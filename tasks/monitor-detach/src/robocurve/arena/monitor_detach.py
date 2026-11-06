"""Monitor detach on yam_arena's station: objects, the fastener and faults, telemetry and scoring.

A yam_arena task file re-exports ``SCENE``, ``OBJECTS``, ``SUCCESS_HOLD_S``, ``success`` and ``score``
from here (``submission/monitor-detach/task.py``). The geometry is ``robocurve.sim.layout.LAYOUT``,
written in the station frame of the rest of this package (origin at the left arm base, z up from the
table top) and placed in yam_arena's world by :data:`STATION`.

Per physics substep (240 Hz) the fastener holds the monitor on its bracket with a stiff 6-DoF spring
until the lever passes the scene's threshold, and in B1 Arm A's grip force fades after the release
(``robocurve.sim.fastener``). After every control step a telemetry record is kept for ``robocurve.scoring``
and appended to a JSONL file for offline re-scoring (``ROBOCURVE_TELEMETRY_DIR``, default ``logs/arena_telemetry``).

The scene variant comes from ``ROBOCURVE_SCENE`` (``B2`` by default, the scored one; ``A``, ``B1``).
Pure Python at import time.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from yam_arena.task_file import TABLE_TOP, Box

from robocurve.arena.hooks import UsdFixture, install, on_reset, on_step, on_substep
from robocurve.scoring import goal_reached, screen_up_deg, trial_score, unplanned_tilt_deg
from robocurve.sim.fastener import SCENES, Fastener, quat_xyzw_to_matrix
from robocurve.sim.layout import LAYOUT, box_center, box_size

install()

ROOT = Path(__file__).resolve().parents[3]
SCENE = "station"
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"
VARIANT = os.environ.get("ROBOCURVE_SCENE", "B2")
assert VARIANT in SCENES, f"ROBOCURVE_SCENE must be one of {sorted(SCENES)}, got {VARIANT!r}"
# B1's reaction window as designed: Arm A alone loses the monitor about 1.8 s after the release. yam_arena's pads
# hold less than yam_sim's (A slips at ~47 % grip, 1.2 s into a 2 s fade), so the fade takes 3 s here.
SCENE_SPECS = {**SCENES, "B1": replace(SCENES["B1"], fade_duration_s=3.0)}
STATION = np.array([0.0, 0.305, TABLE_TOP[SCENE]])  # the station frame's origin in yam_arena's world
ARM_A, ARM_B = "left", "right"
SIDES = ("left", "right")
PHYSICS_DT = 1.0 / 240.0
GRIP_FORCE_N = 50.0  # yam_arena's gripper effort limit (i2rt caps the grip at 50 N)
TCP_OFFSET = np.array([0.0, 0.0, 0.134])  # tool point in the `gripper` link frame (yam_arena.constants)
FINGER_OPEN = 0.04695
HOLD_GAP = (-0.008, 0.006)  # finger gap minus panel thickness that counts as pinching the panel [m]
HOLD_FORCE_N = 2.0
ASSETS = ROOT / "assets" / "usd" / "robocurve"


def _world(p: Any) -> tuple[float, float, float]:
    return tuple(float(v) for v in np.asarray(p, dtype=float) + STATION)  # type: ignore[return-value]


def _build_stand() -> str:
    from robocurve.sim.stand_usd import write_stand_usd

    return str(write_stand_usd(ASSETS / "arena_stand.usda", LAYOUT, world_anchor=_world((0.0, 0.0, 0.0))))


def _build_monitor() -> str:
    from robocurve.sim.stand_usd import write_monitor_usd

    return str(write_monitor_usd(ASSETS / "arena_monitor.usda", LAYOUT))


MONITOR = UsdFixture(
    "monitor",
    pos=_world(box_center(LAYOUT.panel_box)),
    size=box_size(LAYOUT.panel_box),
    mass=LAYOUT.panel_mass,
    color=(0.07, 0.07, 0.08),
    build=_build_monitor,
)
STAND = UsdFixture(
    "stand",
    pos=_world((0.0, 0.0, 0.0)),
    size=(0.11, 0.20, 0.22),
    color=(0.09, 0.09, 0.10),
    build=_build_stand,
    articulation=True,
    drives=(("lever_joint", LAYOUT.lever_stiffness, 5.0),),
)
MAT = Box("mat", pos=_world(box_center(LAYOUT.mat_box)), size=box_size(LAYOUT.mat_box), color=(0.33, 0.62, 0.55))
OBJECTS = [MONITOR, STAND, MAT]

_STATE: dict[str, Any] = {"armed": False, "records": []}


def _np(x: Any) -> np.ndarray:
    """A ProxyArray, torch tensor or warp array as a float64 numpy array."""
    x = getattr(x, "torch", x)
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x.numpy() if hasattr(x, "numpy") else x, dtype=np.float64)


def _ids(environment: Any) -> dict[str, Any]:
    """Joint and body indices, looked up once per env."""
    cache = _STATE.get("ids")
    if cache is not None and cache["env"] is environment:
        return cache
    base = environment.unwrapped
    out: dict[str, Any] = {"env": environment}
    for side in SIDES:
        robot = base.scene[f"{side}_robot"]
        out[side] = {
            "arm": robot.find_joints([f"joint{i}" for i in range(1, 7)], preserve_order=True)[0],
            "fingers": robot.find_joints(["joint7", "joint8"], preserve_order=True)[0],
            "gripper": robot.find_bodies("gripper")[0][0],
        }
    out["lever"] = base.scene["stand"].find_joints("lever_joint")[0][0]
    _STATE["ids"] = out
    return out


@on_reset
def _reset(environment: Any) -> None:
    """Re-latch the fastener and re-arm the faults; the first substep puts the monitor back on its bracket."""
    spec = SCENE_SPECS[VARIANT]
    rest = np.array(MONITOR.pos)
    _STATE.update(
        armed=True,
        fresh=True,
        k=0,
        t0=None,
        spec=spec,
        fastener=Fastener(rest_pos=rest, rest_quat=np.array([0.0, 0.0, 0.0, 1.0]), mass=LAYOUT.panel_mass,
                          threshold=spec.latch_threshold),
        grip_scale=None,
        peak_down=0.0,
        events={},
        records=[],
    )
    _open_log()


@on_substep
def _substep(environment: Any) -> None:
    import torch

    st = _STATE
    if not st["armed"]:
        return
    base = environment.unwrapped
    device = base.device
    ids = _ids(environment)
    panel, stand = base.scene["monitor"], base.scene["stand"]
    origin = _np(base.scene.env_origins[0])
    if st["fresh"]:
        # A seeded reset shifts free objects by up to object_jitter; the monitor starts on its bracket.
        pose = torch.tensor([[*(np.array(MONITOR.pos) + origin), 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
        panel.write_root_pose_to_sim_index(root_pose=pose)
        panel.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=device))
        st["fresh"] = False
    t = st["k"] * PHYSICS_DT
    st["k"] += 1
    fastener: Fastener = st["fastener"]
    lever_frac = float(_np(stand.data.joint_pos)[0, ids["lever"]] / LAYOUT.lever_travel)
    if fastener.update_latch(t, lever_frac):
        st["events"]["latch_open"] = t
    pose = _np(panel.data.root_link_pose_w)[0]
    vel = _np(panel.data.root_com_vel_w)[0]
    st["peak_down"] = max(st["peak_down"], float(-vel[2]))
    force, torque = fastener.wrench(pose[:3] - origin, pose[3:], vel[:3], vel[3:])
    panel.permanent_wrench_composer.set_forces_and_torques_index(
        forces=torch.tensor(force, device=device, dtype=torch.float32).view(1, 1, 3),
        torques=torch.tensor(torque, device=device, dtype=torch.float32).view(1, 1, 3),
        is_global=True,
    )
    opened = fastener.opened_at
    scale = st["spec"].grip_scale(None if opened is None else t - opened)
    if st["grip_scale"] is None or abs(scale - st["grip_scale"]) > 1e-3:
        robot = base.scene[f"{ARM_A}_robot"]
        fingers = ids[ARM_A]["fingers"]
        robot.write_joint_effort_limit_to_sim_index(
            limits=torch.full((1, len(fingers)), GRIP_FORCE_N * scale, device=device), joint_ids=fingers
        )
        st["grip_scale"] = scale


def _panel_min_z(pos: np.ndarray, quat: np.ndarray) -> float:
    half = np.array(LAYOUT.panel_size) / 2
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
    return float(np.min((corners @ quat_xyzw_to_matrix(quat).T)[:, 2] + pos[2]))


@on_step
def _step(environment: Any) -> None:
    st = _STATE
    base = environment.unwrapped
    ids = _ids(environment)
    origin = _np(base.scene.env_origins[0])
    if st["t0"] is None:
        st["t0"] = st["k"] * PHYSICS_DT - base.step_dt
    t = st["k"] * PHYSICS_DT - st["t0"]
    panel, stand = base.scene["monitor"], base.scene["stand"]
    pose = _np(panel.data.root_link_pose_w)[0]
    vel = _np(panel.data.root_com_vel_w)[0]
    pos = pose[:3] - origin - STATION  # station frame
    quat = pose[3:]
    rot = quat_xyzw_to_matrix(quat)
    fastener: Fastener = st["fastener"]
    rec: dict[str, Any] = {
        "t": round(t, 5),
        "latch_open": fastener.is_open,
        "lever_frac": float(_np(stand.data.joint_pos)[0, ids["lever"]] / LAYOUT.lever_travel),
        "load_share": fastener.load_share,
        "fastener_force": fastener.last_force.tolist(),
        "panel_pos": pos.tolist(),
        "panel_quat": quat.tolist(),
        "panel_lin_vel": vel[:3].tolist(),
        "panel_ang_vel": vel[3:].tolist(),
        "peak_down_speed": max(st["peak_down"], float(-vel[2])),
        "panel_min_z": _panel_min_z(pos, quat),
        "screen_up_deg": screen_up_deg(quat),
        "unplanned_tilt_deg": unplanned_tilt_deg(quat, [0.0, 0.0, 0.0, 1.0]),
        "grip_scale_a": st["grip_scale"] if st["grip_scale"] is not None else 1.0,
        "arm_a": ARM_A,
        "finger": {},
        "contact_n": {},
        "holding": {},
        "eff": {},
        "q": {},
        "tcp": {},
    }
    st["peak_down"] = 0.0
    half = np.array(LAYOUT.panel_size) / 2
    for side in SIDES:
        robot = base.scene[f"{side}_robot"]
        q = _np(robot.data.joint_pos)[0]
        forces = _np(robot.root_view.get_dof_projected_joint_forces())[0]
        fingers = np.abs(q[ids[side]["fingers"]])
        finger_force = np.abs(forces[ids[side]["fingers"]])
        g = _np(robot.data.body_link_pose_w)[0, ids[side]["gripper"]]
        tcp = g[:3] + quat_xyzw_to_matrix(g[3:]) @ TCP_OFFSET - origin - STATION
        local = rot.T @ (tcp - pos)
        on_panel = abs(local[0]) < half[0] + 0.012 and abs(local[1]) < half[1] + 0.01 and abs(local[2]) < half[2] + 0.01
        gap = float(fingers.sum()) - LAYOUT.panel_size[0]
        rec["finger"][side] = float(fingers.mean())
        rec["contact_n"][f"{side}_tip_left"], rec["contact_n"][f"{side}_tip_right"] = (float(f) for f in finger_force)
        rec["holding"][side] = bool(on_panel and HOLD_GAP[0] < gap < HOLD_GAP[1] and finger_force.min() > HOLD_FORCE_N)
        rec["eff"][side] = forces[ids[side]["arm"]].tolist()
        rec["q"][side] = q[ids[side]["arm"]].tolist()
        rec["tcp"][side] = tcp.tolist()
    st["records"].append(rec)
    log = st.get("log")
    if log is not None:
        log.write(json.dumps(rec) + "\n")
        log.flush()


def _open_log() -> None:
    old = _STATE.get("log")
    if old is not None:
        old.close()
    folder = Path(os.environ.get("ROBOCURVE_TELEMETRY_DIR", "logs/arena_telemetry"))
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{time.strftime('%Y%m%dT%H%M%S')}_{VARIANT}.jsonl"
    _STATE["log"] = open(path, "w")
    _STATE["log_path"] = path
    _STATE["log"].write(json.dumps({"scene": VARIANT, "scene_spec": VARIANT, "arm_a": ARM_A, "embodiment": "yam_arena"}) + "\n")


def records() -> list[dict[str, Any]]:
    """This episode's telemetry so far (station frame)."""
    return _STATE["records"]


def events() -> dict[str, float]:
    """Named event times of this episode, on the telemetry's clock."""
    t0 = _STATE.get("t0") or 0.0
    return {k: v - t0 for k, v in _STATE.get("events", {}).items()}


def success(objects: Any) -> bool:
    """Standing upright at rest on the mat, held by both grippers (robocurve.scoring.goal_reached)."""
    recs = _STATE["records"]
    return bool(recs) and goal_reached(recs, LAYOUT, ARM_A, ARM_B)


def score(objects: Any) -> float:
    """The benchmark score so far: 0, 0.25, 0.5 or 0.75 (1.0 is success); robocurve.scoring.trial_score."""
    recs = _STATE["records"]
    return trial_score(recs, LAYOUT, ARM_A, ARM_B)[0] if recs else 0.0
