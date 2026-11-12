"""On the station table, in front of the robot: a potted rose bush with four green, leafy canes and, at the front of the pot, one
dead cane. The dead cane is green from the soil up to just above a small red bud and brown above, up to an old pruning cut. Bypass
secateurs with red handles lie open on the table to the right of the pot, and an open black garden tub stands to the left.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \\
           -E cam_width=320 -E cam_height=176 --scorer task_score --max-steps 1200 --instruction "<the task.json instruction>"
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import scene_assets as sa  # noqa: E402
import scenekit  # noqa: E402
import steptime  # noqa: E402
from motion_checks import points_at_rest  # noqa: E402
from pruning_checks import PruningState  # noqa: E402
from cut_checks import footprint_distance, section_contact  # noqa: E402
from scenekit import UsdBody, on_reset, on_step  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP, TCP_OFFSET  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

sa.write_all()

_bud_mesh = sa.bud_mesh()
BUD_VERTICES = _bud_mesh.v
BUD_EDGES = np.unique(np.sort(np.concatenate([_bud_mesh.f[:, [0, 1]], _bud_mesh.f[:, [1, 2]],
                                              _bud_mesh.f[:, [2, 0]]]), axis=1), axis=0)
BUD_EDGES = BUD_EDGES[np.linalg.norm(BUD_VERTICES[BUD_EDGES[:, 0]] - BUD_VERTICES[BUD_EDGES[:, 1]], axis=1) > 1e-8]
_BLADE_MESH = next(mesh for name, mesh, _ in sa._closed_half(1)[0] if name == "blade")

CHAIN = ("c_px_joint", "c_py_joint", "c_pz_joint", "c_rx_joint", "c_ry_joint", "c_rz_joint")
SLIDES = ("stub_green_slide", "stub_brown_slide", "stub_bud_slide", "cut_brown_slide", "cut_green_slide", "cut_bud_slide")
CHAIN_GAINS = {"c_p": (3.0e4, 100.0), "c_r": (50.0, 1.0)}  # stiffness, damping: they hold the cutting in place until the cut
BUSH = UsdBody("bush", usd="assets/bush.usda", pos=(sa.POT_XY[0], sa.POT_XY[1], TOP), size=(0.24, 0.24, 0.33), articulation=True,
               joint_pos=((".*", 0.0),),
               joint_groups=(("stub_.*", 200.0, 5.0e4), ("c_p.*", CHAIN_GAINS["c_p"][1], CHAIN_GAINS["c_p"][0]),
                             ("c_r.*", CHAIN_GAINS["c_r"][1], CHAIN_GAINS["c_r"][0]), ("cut_.*", 50.0, 2.0e4)),
               joint_armature=1.0e-5, friction=(0.9, 0.8), solver_iters=32)
SECATEURS = UsdBody("secateurs", usd="assets/secateurs.usda", pos=(sa.SECATEURS_START[0], sa.SECATEURS_START[1], TOP + sa.TOOL_HALF_T + 0.0005),
                    size=(0.08, 0.20, 0.02), articulation=True, joint_pos=(("pivot", 0.0),), joint_groups=(("pivot", 0.02, sa.SPRING_K),),
                    joint_armature=1.0e-5, friction=(1.0, 0.9), solver_iters=32)
BIN = UsdBody("bin", usd="assets/bin.usda", pos=(sa.BIN_XY[0], sa.BIN_XY[1], TOP), size=(0.27, 0.27, 0.12), mass=0.6, friction=(0.6, 0.5))
OBJECTS = [BUSH, SECATEURS, BIN]

GOOD_CUT = (0.003, 0.015)  # a good cut leaves the stub's top this far above the bud's top
FALL_DROP = 0.03           # a free cutting dropping this far, away from the bin's opening, has fallen
POT_MOVED, POT_TILT_DEG = 0.02, 10.0
CLEAR = 0.05               # every fingertip this far from the secateurs, the canes and the cutting at the end (from their surfaces)
TOOL_HALF_W = 0.008        # the secateurs' blades and handles, as lines this far from their surface
REST_MOVE, REST_STEPS = 0.005, 20  # the cutting moves less than this over the last second
BUD_ATTACH_TOL = 0.003     # the original bud's body stays within 3 mm of its original place on the pot
PAD_MID = np.array([0.0, 0.0, TCP_OFFSET[2] + PAD_TIP_PAST_TCP - 0.012])  # a finger pad's middle, in the gripper frame
PAD_TIP = np.array([0.0, 0.0, TCP_OFFSET[2] + PAD_TIP_PAST_TCP])

_S: dict = {}


def _clear() -> None:
    _S.clear()
    _S.update(ready=False, cut=False, h=None, green_cut=False, fell=False, pot_moved=False, tool_lost=False, in_jaw={}, closed=False, drop_ref=None,
              history=[], held=False, in_bin=False, over_bin=False, tool_down=False, clear=False, t=0,
              pruning=PruningState(sa.BUD_TOP_S), stub_height=sa.TOP_S, bud_attached=True, bud_cut_separately=False, cut_events=[], full_conditions=[], last_pivot=None)


def _np(x) -> np.ndarray:
    return getattr(x, "torch", x).detach().cpu().numpy().astype(np.float64)


def _rotation(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _body(asset, name: str, origin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    i = asset.body_names.index(name)
    pose = _np(asset.data.body_link_pose_w)[0, i]
    return pose[:3] - origin - np.array([0.0, 0.0, TOP]), _rotation(pose[3:7])


def _pads(env, origin: np.ndarray) -> dict[str, tuple[np.ndarray, float]]:
    """Each arm's two finger-pad middles and two pad tips (station frame, heights above the table), and its finger gap."""
    base = env.unwrapped
    out = {}
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        i = robot.body_names.index("gripper")
        pos = _np(robot.data.body_pos_w)[0, i] - origin - np.array([0.0, 0.0, TOP])
        rot = _rotation(_np(robot.data.body_quat_w)[0, i])
        half = abs(float(scenekit.arm_joints(env, side, ["joint7"])[0]))
        pts = [pos + rot @ (ref + np.array([s * half, 0.0, 0.0])) for ref in (PAD_MID, PAD_TIP) for s in (-1.0, 1.0)]
        out[side] = (np.array(pts), 2 * half)
    return out


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    d = b - a
    t = float(np.clip(np.dot(p - a, d) / max(np.dot(d, d), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * d)))


def _write_joints(asset, values: dict[str, float]) -> None:
    import torch

    ids = [asset.joint_names.index(n) for n in values]
    pos = torch.tensor([list(values.values())], device=asset.device, dtype=torch.float32)
    asset.write_joint_position_to_sim_index(position=pos, joint_ids=ids)
    asset.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(pos), joint_ids=ids)
    asset.set_joint_position_target_index(target=pos, joint_ids=ids)


def _chain_gains(asset, on: bool) -> None:
    import torch

    for prefix, (k, c) in CHAIN_GAINS.items():
        ids = [i for i, n in enumerate(asset.joint_names) if n.startswith(prefix)]
        stiff = torch.full((1, len(ids)), k if on else 0.0, device=asset.device, dtype=torch.float32)
        damp = torch.full((1, len(ids)), c if on else 1.0e-4, device=asset.device, dtype=torch.float32)
        asset.write_joint_stiffness_to_sim_index(stiffness=stiff, joint_ids=ids)
        asset.write_joint_damping_to_sim_index(damping=damp, joint_ids=ids)


def _spring(tool) -> None:
    """The secateurs' spring, preloaded against the open stop."""
    import torch

    i = tool.joint_names.index("pivot")
    tool.set_joint_position_target_index(target=torch.tensor([[-sa.SPRING_PRELOAD]], device=tool.device, dtype=torch.float32), joint_ids=[i])


def _restore(env) -> None:
    base = env.unwrapped
    bush = base.scene["bush"]
    _write_joints(bush, {n: 0.0 for n in CHAIN + SLIDES})
    _chain_gains(bush, True)
    _spring(base.scene["secateurs"])


def _retained_plant(bush, origin: np.ndarray) -> tuple[float, bool]:
    """The visible stub's length and the original bud's attachment, from actual link poses."""
    pot, pot_rot = _body(bush, "pot", origin)
    base = pot + pot_rot @ sa.on_cane(0.0)
    axis = pot_rot @ sa.CANE_DIR
    ends = []
    for name, length in (("stub_green", sa.L_SG), ("stub_brown", sa.L_SB)):
        pos, rot = _body(bush, name, origin)
        ends.append(float(axis @ (pos + rot @ np.array([0.0, 0.0, length]) - base)))
    bud, _ = _body(bush, "stub_bud", origin)
    expected = pot + pot_rot @ sa.on_cane(sa.BUD_S)
    height = max(0.006, *ends)
    attached = float(np.linalg.norm(bud - expected)) <= BUD_ATTACH_TOL and height >= sa.BUD_TOP_S
    return height, attached


def _canes(bush, origin: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray, float]]:
    """All remaining cut targets, including the retained stub and its original bud."""
    pos, rot = _body(bush, "pot", origin)
    out = []
    if not _S["cut"]:
        out.append(("dead", pos + rot @ sa.on_cane(0.0), pos + rot @ sa.on_cane(sa.TOP_S), 0.0))
    else:
        height, _ = _retained_plant(bush, origin)
        height = min(height, _S["pruning"].stub_length)
        out.append(("stub", pos + rot @ sa.on_cane(0.0), pos + rot @ sa.on_cane(height), 0.0))
    if not _S["pruning"].bud_damaged:
        bud, bud_rot = _body(bush, "stub_bud", origin)
        vertices = bud + BUD_VERTICES @ bud_rot.T
        # A blade plane intersects the actual bud mesh, regardless of its approach direction.
        for k, (a, b) in enumerate(BUD_EDGES):
            out.append((f"bud{k}", vertices[a], vertices[b], 0.0))
    for k, (b, t, _r) in enumerate(sa.GREEN_CANES):
        base = pos + rot @ np.array([b[0], b[1], sa.SOIL_Z])
        out.append((f"green{k}", base, pos + rot @ np.asarray(t, float), sa.GREEN_CUT_FROM - sa.SOIL_Z))
    return out


def _blade_frame(point, q):
    """A hook-frame plane point in the blade mesh's authored closed frame."""
    angle = q - sa.HALF
    c, sn = math.cos(angle), math.sin(angle)
    return np.array([c * point[0] + sn * point[1], -sn * point[0] + c * point[1]])


def _check_cut(env, origin: np.ndarray) -> None:
    """Record closure only at current finite blade/material contact, never remembered jaw entry."""
    base = env.unwrapped
    tool, bush = base.scene["secateurs"], base.scene["bush"]
    o, r = _body(tool, "hook_half", origin)
    q = float(_np(tool.data.joint_pos)[0, tool.joint_names.index("pivot")])
    half_open = (sa.JAW_OPEN - q) / 2
    closed = q >= sa.JAW_OPEN - 0.01
    closing_edge = closed and not _S["closed"]
    closing = _S["last_pivot"] is not None and q > _S["last_pivot"] + 1e-6
    n = r[:, 2]
    bud_section = []
    for name, a, b, s_min in _canes(bush, origin):
        d = b - a
        length = float(np.linalg.norm(d))
        d = d / length
        is_bud = name.startswith("bud")
        _S["in_jaw"][name] = False  # diagnostic only; never authorizes a cut
        if abs(float(n @ d)) < (1e-8 if is_bud else 0.3):
            continue
        s = (sa.BLADE_Z - float(n @ (a - o))) / float(n @ d)
        if not s_min <= s <= length:
            continue
        point = a + s * d
        local = r.T @ (point - o)
        if is_bud:
            bud_section.append(_blade_frame(local, q))
            continue
        radius = math.hypot(local[0], local[1])
        reach = sa.JAW_REACH[0] <= radius <= sa.JAW_REACH[1]
        phi_mid = math.atan2(-local[0], local[1]) - q / 2
        _S["in_jaw"][name] = reach and abs(phi_mid) <= half_open
        if name in ("dead", "stub"):
            material_radius = sa.R1 if sa.BROWN_S <= s < sa.T_LOW_S else sa.R0
        else:
            material_radius = sa.GREEN_CANES[int(name[5:])][2]
        distance = footprint_distance(_blade_frame(local, q), _BLADE_MESH.v, _BLADE_MESH.f) if closing_edge and reach else float("inf")
        # A radius-R disk is inside the oblique cylinder's elliptical section:
        # this contact gate is conservative, without enlarging the visible cane.
        if distance > material_radius:
            continue
        _S["cut_events"].append({"target": name, "height": float(s), "t": _S["t"] / 20.0,
                                 "contact": "blade_mesh", "blade_distance": distance, "material_radius": material_radius,
                                 "point": point.tolist(), "q": q})
        if name == "dead":
            _S["pruning"].cut(s)
            values = sa.cut_joint_positions(s)
            _S["h"] = min(max(s, sa.H_MIN_S), sa.T_LOW_S - 0.003)
            _S["cut"] = True
            values.update(sa.stub_joint_positions(s, not _S["pruning"].bud_damaged))
            if _S["bud_cut_separately"]:
                values["cut_bud_slide"] = 0.0
            _write_joints(bush, values)
            _chain_gains(bush, False)
        elif name == "stub":
            _S["pruning"].cut(s)
            _write_joints(bush, sa.stub_joint_positions(_S["pruning"].stub_length, not _S["pruning"].bud_damaged))
        else:
            _S["green_cut"] = True
    if closing and not _S["pruning"].bud_damaged:
        contact = section_contact(bud_section, _BLADE_MESH.v, _BLADE_MESH.f, sa.JAW_REACH)
        if contact is not None:
            angle = q - sa.HALF
            c, sn = math.cos(angle), math.sin(angle)
            local = np.array([c * contact[0] - sn * contact[1], sn * contact[0] + c * contact[1], sa.BLADE_Z])
            _S["cut_events"].append({"target": "bud", "height": None, "t": _S["t"] / 20.0,
                                     "contact": "blade_mesh_section", "point": (o + r @ local).tolist(), "q": q})
            _S["pruning"].bud_damaged = True
            _S["bud_cut_separately"] = True
            _write_joints(bush, {"stub_bud_slide": sa.BUD_SINK})
    _S["closed"] = closed
    _S["last_pivot"] = q


def _cutting(bush, origin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos, rot = _body(bush, "cut_top", origin)
    return pos + rot @ np.array([0.0, 0.0, _S["h"] - sa.T_LOW_S]), pos + rot @ np.array([0.0, 0.0, sa.L_T])


def _tool_points(tool, origin: np.ndarray) -> np.ndarray:
    """Points along each half of the secateurs: the pivot, the jaw's tip, the handle's middle and its end (station frame)."""
    pts = []
    for half, side in (("hook_half", -1.0), ("blade_half", 1.0)):
        o, r = _body(tool, half, origin)
        angle = -side * sa.HALF  # each half's turn from closed to open, built into its link
        c, s = math.cos(angle), math.sin(angle)
        for x, y in ((0.0, 0.0), (side * 0.004, 0.05), (-side * sa.HANDLE_X, -0.05), (-side * sa.HANDLE_X, -0.12)):
            pts.append(o + r @ np.array([c * x - s * y, s * x + c * y, 0.0]))
    return np.array(pts)


def _measure(env) -> None:
    base = env.unwrapped
    if not _S.get("ready"):
        _clear()
        _restore(env)
        _S["ready"] = True
    origin = _np(base.scene.env_origins[0])
    bush, tool, tub = base.scene["bush"], base.scene["secateurs"], base.scene["bin"]
    _S["t"] += 1
    steptime.tick()

    pot, pot_rot = _body(bush, "pot", origin)
    moved = float(np.linalg.norm(pot - np.array([sa.POT_XY[0], sa.POT_XY[1], 0.0])))
    tilt = math.degrees(math.acos(float(np.clip(pot_rot[2, 2], -1.0, 1.0))))
    _S["pot_moved"] = _S["pot_moved"] or moved > POT_MOVED or tilt > POT_TILT_DEG
    tool_o, tool_r = _body(tool, "hook_half", origin)
    _S["tool_lost"] = _S["tool_lost"] or tool_o[2] < -0.05 or abs(tool_o[0] - 0.36) > 0.55 or abs(tool_o[1]) > 0.65

    _check_cut(env, origin)
    stub_height, bud_attached = _retained_plant(bush, origin)
    _S["pruning"].observe_bud(bud_attached)
    _S.update(stub_height=stub_height, bud_attached=bud_attached)
    pads = _pads(env, origin)
    tub_pose = _np(tub.data.root_link_pose_w)[0]
    tub_pos = tub_pose[:3] - origin - np.array([0.0, 0.0, TOP])
    tub_upright = _rotation(tub_pose[3:7])[2, 2] > math.cos(math.radians(30.0))  # a tipped tub's opening faces sideways
    tool_pts = _tool_points(tool, origin)

    cane_segments = []
    stub_top = stub_height if _S["cut"] else sa.TOP_S
    cane_segments.append((pot + pot_rot @ sa.on_cane(0.0), pot + pot_rot @ sa.on_cane(stub_top), sa.R0))
    for b, t, r in sa.GREEN_CANES:
        cane_segments.append((pot + pot_rot @ np.array([b[0], b[1], sa.SOIL_Z]), pot + pot_rot @ np.asarray(t, float), r))
    p0, p1, p2, p3 = tool_pts[0], tool_pts[1], tool_pts[2], tool_pts[3]  # pivot, jaw, handle middle, handle end (hook half)
    q0, q1, q2, q3 = tool_pts[4], tool_pts[5], tool_pts[6], tool_pts[7]  # ... and the blade half
    tool_segments = [(p0, p1, TOOL_HALF_W), (p0, p2, TOOL_HALF_W), (p2, p3, TOOL_HALF_W),
                     (q0, q1, TOOL_HALF_W), (q0, q2, TOOL_HALF_W), (q2, q3, TOOL_HALF_W)]

    near = []
    if _S["cut"]:
        bottom, top = _cutting(bush, origin)
        held = False
        for pts, gap in pads.values():
            mids = pts[:2]
            if gap <= 2 * sa.R0 + 0.006 and all(_seg_dist(p, bottom, top) <= sa.R0 + 0.004 for p in mids):
                held = True
        mid = (bottom + top) / 2
        inner = sa.BIN_R_BOTTOM - sa.BIN_WALL + (sa.BIN_R_TOP - sa.BIN_R_BOTTOM) * 0.5
        over_bin = tub_upright and math.hypot(mid[0] - tub_pos[0], mid[1] - tub_pos[1]) <= inner
        low = min(bottom[2], top[2])
        in_bin = (not held) and over_bin and low < tub_pos[2] + sa.BIN_H
        ends = np.array([bottom[2], top[2]])
        if held or over_bin:
            _S["drop_ref"] = None
        else:
            ref = ends if _S["drop_ref"] is None else np.maximum(_S["drop_ref"], ends)
            _S["drop_ref"] = ref
            if float(np.max(ref - ends)) >= FALL_DROP:
                _S["fell"] = True
        _S["history"] = (_S["history"] + [np.concatenate([bottom, top])])[-(REST_STEPS + 1):]
        _S.update(held=held, in_bin=in_bin, over_bin=over_bin)
        near.append((bottom, top, sa.R0))

    tips = np.vstack([pts for pts, _ in pads.values()])
    gap = float("inf")
    for p in tips:
        for a, b, radius in cane_segments + near + tool_segments:
            gap = min(gap, _seg_dist(p, a, b) - radius)
    tilt_tool = math.degrees(math.acos(float(np.clip(abs(tool_r[2, 2]), -1.0, 1.0))))
    off_pot = all(math.hypot(p[0] - pot[0], p[1] - pot[1]) > sa.POT_R_TOP + 0.005 for p in tool_pts)
    _S["tool_down"] = tool_o[2] < sa.TOOL_HALF_T + 0.012 and tilt_tool < 20.0 and off_pot
    _S["clear"] = gap >= CLEAR
    _record_full_conditions()
    _log(tool, bush, tool_o, tool_r, pot, pads, gap)


def _log(tool, bush, tool_o, tool_r, pot, pads, gap) -> None:
    """With TASK_SCORE_DIR set (the run scripts set it), one line of state every 5 control steps, for debugging."""
    folder = os.environ.get("TASK_SCORE_DIR")
    if not folder or _S["t"] % 5:
        return
    import json

    os.makedirs(folder, exist_ok=True)
    q = float(_np(tool.data.joint_pos)[0, tool.joint_names.index("pivot")])
    row = {"t": _S["t"] / 20.0, "jaw": round(q, 4), "tool": [round(float(v), 4) for v in tool_o], "tool_up": round(float(tool_r[2, 2]), 3),
           "pot": [round(float(v), 4) for v in pot], "gap": round(float(gap), 3),
           "grips": {k: round(g, 4) for k, (_, g) in pads.items()}, "tcp": {k: [round(float(v), 4) for v in p[:2].mean(axis=0)] for k, (p, _) in pads.items()},
           **{k: bool(_S[k]) for k in ("cut", "green_cut", "fell", "pot_moved", "tool_lost", "held", "in_bin", "over_bin", "tool_down", "clear")}}
    if _S["h"] is not None:
        row["h"] = round(_S["h"], 4)
        row["cutting"] = [round(float(v), 4) for v in _S["history"][-1]] if _S["history"] else None
    row.update(stub_height=round(_S["stub_height"], 4), bud_attached=bool(_S["bud_attached"]),
               bud_damaged=bool(_S["pruning"].bud_damaged), cuts=_S["pruning"].cuts,
               first_cut=_S["pruning"].first_cut, cut_events=_S["cut_events"],
               full_condition_samples=len(_S["full_conditions"]), full_condition_good=sum(_S["full_conditions"]),
               tool_yaw=round(math.atan2(tool_r[1, 0], tool_r[0, 0]), 4))
    joints = _np(bush.data.joint_pos)[0]
    row["sleeve_joints"] = {name: round(float(joints[bush.joint_names.index(name)]), 6) for name in SLIDES}
    with open(os.path.join(folder, "state.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")


@on_reset
def _on_reset(env) -> None:
    _clear()
    _restore(env)
    _S["ready"] = True


@on_step
def _on_step(env) -> None:
    _measure(env)


def _at_rest() -> bool:
    return points_at_rest(_S["history"], REST_MOVE, REST_STEPS)


def good_cut() -> bool:
    return _S["pruning"].credit(_S["stub_height"], _S["bud_attached"], GOOD_CUT) == 1.0


def _full_conditions_now() -> bool:
    """Instantaneous necessary conditions; motion uses the same final one-second window separately."""
    return bool(_S and _S["cut"] and not any(_S[k] for k in ("green_cut", "fell", "pot_moved", "tool_lost"))
                and good_cut() and _S["in_bin"] and _S["tool_down"] and _S["clear"])


def _record_full_conditions() -> None:
    """Called once by the control-step measurement hook, never by a score query."""
    _S["full_conditions"] = (_S["full_conditions"] + [_full_conditions_now()])[-(REST_STEPS + 1):]


def score(objects) -> float:
    """0 for a cut or detached original bud, a green cut, a fall, a moved pot, lost secateurs, or no cut. Otherwise 0.25 if the
    first cut or final stub is misplaced; 0.5 if both are good but the cutting is outside the bin; 0.75 if it is in the bin;
    1.0 also requires the secateurs down outside the pot, fingertips clear, and the cutting still."""
    if not _S or _S["green_cut"] or _S["fell"] or _S["pot_moved"] or _S["tool_lost"] or not _S["cut"]:
        return 0.0
    plant_credit = _S["pruning"].credit(_S["stub_height"], _S["bud_attached"], GOOD_CUT)
    if plant_credit < 1.0:
        return plant_credit
    if not _S["in_bin"]:
        return 0.5
    complete_window = len(_S["full_conditions"]) == REST_STEPS + 1 and all(_S["full_conditions"])
    return 1.0 if complete_window and _full_conditions_now() and _at_rest() else 0.75


def success(objects) -> bool:
    return score(objects) == 1.0


# -- the scripted demo -------------------------------------------------------------------------------------------------

YAW_RIGHT = math.pi / 2  # fingers along x, across the secateurs' handles
YAW_LEFT = -math.pi / 2  # fingers along x too, across the leaning cane
TCP_OVER_TOOL = 0.007    # the right TCP above the tool's middle when it grips the handles
GRIP_BACK = 0.030        # ... and this far behind the pivot (further back, the open handles splay wider than the open fingers)
CUT_RADIUS = 0.030       # the cane crosses the jaws this far from the pivot
TOOL_SAG = 0.008         # held by its handles, the tool hangs this much lower than the grip (measured)
CUT_AT = sa.BUD_TOP_S + 0.009   # along the cane: 9 mm above the bud
HOLD_AT = 0.205          # the left hand holds the cane here, near its tip
PINCH_BELOW_TCP = 0.006  # the cane sits this far below the left TCP, between the pads
LEFT_CLEAR = (0.15, 0.20, TOP + 0.20)
RIGHT_CLEAR = (0.14, -0.305, TOP + 0.16)


def _handle_outer_x(y_tool: float) -> float:
    """At rest, the outer face of the hook half's handle at y (tool frame)."""
    xo, c, s = sa.HANDLE_X + sa.HANDLE_W / 2, math.cos(sa.HALF), math.sin(sa.HALF)
    y_c = (y_tool - xo * s) / c
    return xo * c - y_c * s


def _cane_point(s: float) -> np.ndarray:
    """A point on the dead cane's axis, s along it from the soil, in the station frame (height above the table)."""
    return sa.on_cane(s) + np.array([sa.POT_XY[0], sa.POT_XY[1], 0.0])


def demo() -> dict[str, list[Waypoint]]:
    """TASK_DEMO=failure: the left arm does not hold the cane, so the cutting falls. TASK_DEMO=brown: the cut is made far out, in
    the dead wood. recut/side_bud cut the retained plant; retrim stays above the bud; repair starts with a misplaced cut.
    Open loop, with timed waits only."""
    mode = os.environ.get("TASK_DEMO", "success")
    cut = _cane_point(0.083 if mode == "repair" else 0.090 if mode == "brown" else CUT_AT)
    held = _cane_point(HOLD_AT)
    px, py = sa.SECATEURS_START
    grip_y = py - GRIP_BACK
    hold = -(_handle_outer_x(-GRIP_BACK) - 0.003)  # squeeze 3 mm a side: the spring then holds the tool against the pads
    z_table = TOP + sa.TOOL_HALF_T + 0.0005 + TCP_OVER_TOOL
    z_cut = TOP + cut[2] - sa.BLADE_Z + TCP_OVER_TOOL + TOOL_SAG
    cut_x, cut_y = cut[0], cut[1] - CUT_RADIUS - GRIP_BACK
    left_z = TOP + held[2] + PINCH_BELOW_TCP
    down = (0.20, -0.18)
    right = [
        Waypoint(1.5, (px, grip_y, TOP + 0.08), FINGER_OPEN, "over the secateurs", YAW_RIGHT),
        Waypoint(1.2, (px, grip_y, z_table), FINGER_OPEN, "round the handles", YAW_RIGHT),
        Waypoint(0.8, (px, grip_y, z_table), hold, "hold the handles", YAW_RIGHT),
        Waypoint(1.2, (px, grip_y, z_cut), hold, "lift", YAW_RIGHT),
        Waypoint(1.6, (cut_x, cut_y - 0.06, z_cut), hold, "jaws towards the dead cane", YAW_RIGHT),
        Waypoint(1.4, (cut_x, cut_y, z_cut), hold, "jaws round it, above the bud", YAW_RIGHT),
    ]
    t_ready = sum(w.duration for w in right)
    left: list[Waypoint] = []
    if mode != "failure":
        left = [
            Waypoint(1.6, (held[0], held[1] + 0.07, left_z), FINGER_OPEN, "beyond the cane's tip", YAW_LEFT),
            Waypoint(1.4, (held[0], held[1], left_z), FINGER_OPEN, "round the cane near its tip", YAW_LEFT),
            Waypoint(0.8, (held[0], held[1], left_z), FINGER_CLOSED, "hold it", YAW_LEFT),
        ]
    t_held = sum(w.duration for w in left)
    t_cut0 = max(t_ready, t_held) + 0.4
    right.append(Waypoint(t_cut0 - t_ready, (cut_x, cut_y, z_cut), hold, "wait", YAW_RIGHT))
    right += [
        Waypoint(1.0, (cut_x, cut_y, z_cut), FINGER_CLOSED, "squeeze: cut", YAW_RIGHT),
        Waypoint(0.8, (cut_x, cut_y, z_cut), FINGER_CLOSED, "hold", YAW_RIGHT),
        Waypoint(0.8, (cut_x, cut_y, z_cut), hold, "let the spring open the jaws", YAW_RIGHT),
    ]
    t_open = sum(w.duration for w in right)
    if mode in ("recut", "side_bud", "retrim", "repair"):
        later_s = {"recut": 0.050, "side_bud": sa.BUD_S - 0.007, "retrim": 0.062, "repair": 0.065}[mode]
        later = _cane_point(later_s)
        later_z = TOP + later[2] - sa.BLADE_Z + TCP_OVER_TOOL + TOOL_SAG
        later_x = later[0] - (0.010 if mode == "side_bud" else 0.0)
        later_y = later[1] - CUT_RADIUS - GRIP_BACK
        right += [
            Waypoint(1.4, (cut_x, cut_y - 0.07, z_cut), hold, "back out of the first cut", YAW_RIGHT),
            Waypoint(1.0, (later_x, later_y - 0.05, later_z), hold, "beside the retained stub", YAW_RIGHT),
            Waypoint(1.2, (later_x, later_y, later_z), hold, "jaws round the retained plant", YAW_RIGHT),
            Waypoint(1.0, (later_x, later_y, later_z), FINGER_CLOSED, "second cut", YAW_RIGHT),
            Waypoint(0.6, (later_x, later_y, later_z), FINGER_CLOSED, "hold", YAW_RIGHT),
            Waypoint(0.8, (later_x, later_y, later_z), hold, "open the jaws", YAW_RIGHT),
        ]
        cut_x, cut_y, z_cut = later_x, later_y, later_z
    right += [
        Waypoint(1.4, (cut_x, cut_y - 0.07, z_cut), hold, "back out", YAW_RIGHT),
        Waypoint(1.5, (down[0], down[1], TOP + 0.07), hold, "to a free spot", YAW_RIGHT),
        Waypoint(1.0, (down[0], down[1], z_table + 0.002), hold, "lay them down", YAW_RIGHT),
        Waypoint(0.6, (down[0], down[1], z_table + 0.002), FINGER_OPEN, "let go", YAW_RIGHT),
        Waypoint(0.8, (down[0], down[1], TOP + 0.09), FINGER_OPEN, "rise", YAW_RIGHT),
        Waypoint(1.2, RIGHT_CLEAR, FINGER_OPEN, "clear", 0.0),
        Waypoint(3.0, RIGHT_CLEAR, FINGER_OPEN, "watch", 0.0),
    ]
    if mode == "failure":
        return {"right": right}
    bx, by = sa.BIN_XY
    centre_from_hand = (HOLD_AT - (CUT_AT + sa.TOP_S) / 2) * sa.CANE_DIR  # from the cutting's middle to the hand
    over = (bx + centre_from_hand[0], by + centre_from_hand[1])
    left += [
        Waypoint(max(t_open - t_held, 0.1), (held[0], held[1], left_z), FINGER_CLOSED, "hold the cutting", YAW_LEFT),
        Waypoint(1.2, (held[0] - 0.04, held[1] + 0.03, TOP + 0.20), FINGER_CLOSED, "lift it clear", YAW_LEFT),
        Waypoint(1.5, (over[0], over[1], TOP + 0.20), FINGER_CLOSED, "over the bin", YAW_LEFT),
        Waypoint(1.2, (over[0], over[1], TOP + 0.15), FINGER_CLOSED, "lower it in", YAW_LEFT),
        Waypoint(0.6, (over[0], over[1], TOP + 0.15), FINGER_OPEN, "let go", YAW_LEFT),
        Waypoint(0.8, (over[0], over[1], TOP + 0.20), FINGER_OPEN, "rise", YAW_LEFT),
        Waypoint(1.2, LEFT_CLEAR, FINGER_OPEN, "clear", YAW_LEFT),
        Waypoint(3.0, LEFT_CLEAR, FINGER_OPEN, "watch", YAW_LEFT),
    ]
    return {"left": left, "right": right}
