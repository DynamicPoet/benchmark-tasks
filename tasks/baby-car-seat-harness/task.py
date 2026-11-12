"""On the station table, between the arms: a rear-facing infant car seat, reclined on its base, its foot end towards the robot. A
newborn doll lies in it, head away from the robot. The harness is undone: each strap comes out of a slot beside the doll's shoulder
and ends in a tongue parked in a pocket beside the doll's hip. The crotch buckle stands between the doll's legs, its two slots
facing up. The red chest clip is open, its two halves over the doll's belly.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \\
           -E cam_width=320 -E cam_height=176 --scorer task_score --max-steps 1600 --instruction "<the task.json instruction>"
"""

from __future__ import annotations

import math
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import scene_assets as sa  # noqa: E402
import scenekit  # noqa: E402
import steptime  # noqa: E402
import harness_fit as hf  # noqa: E402
from scenekit import UsdBody, on_reset, on_step  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP, TCP_OFFSET  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402
from yam_arena import station as _station  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

sa.write_all()

SEAT = UsdBody("seat", usd="assets/seat.usda", pos=(0.0, 0.0, TOP), size=(0.4, 0.3, 0.2), friction=(0.6, 0.5))
HARNESS = UsdBody("harness", usd="assets/harness.usda", pos=(0.0, 0.0, TOP), size=(0.2, 0.2, 0.1), articulation=True,
                  joint_pos=(("(?!fit_fold_[lr]$).*", 0.0), ("fit_fold_l", sa.initial_fold(1)), ("fit_fold_r", sa.initial_fold(-1))),
                  joint_groups=(("strap_.*|tongue_.*", 0.01), ("clip_slide", 3.0), ("clip_half_.*", 1.0), ("fit_.*|adjuster_.*", 0.2)),
                  joint_armature=1.0e-5, friction=(0.8, 0.7), solver_iters=32)
OBJECTS = [SEAT, HARNESS]

LATCH_DEPTH = 0.01    # a tongue latches once its blade is this far into its slot
CLIP_SNAP = 0.005     # the clip snaps shut when its halves are this close
ON_LINE = 0.02        # the closed clip's middle within this of the armpit line
NEAR_HEAD = 0.01      # no fingertip this close to the head
SEAT_MOVED = 0.02
SEAT_TILT_DEG = 5.0
CLEAR = 0.05          # every fingertip this far from the seat and the doll at the end
REST_MOVE = 0.005     # the clip moves less than this ...
REST_STEPS = 20       # ... over the last second at 20 Hz
TIP_IN_GRIPPER = np.array([0.0, 0.0, TCP_OFFSET[2] + PAD_TIP_PAST_TCP])
LOCK_STIFFNESS = {"strap_yaw": 30.0, "strap_pitch": 30.0, "strap_len": 3000.0, "tongue_pitch": 30.0, "tongue_yaw": 30.0, "clip_half": 500.0}
TAGS = {"l": 1, "r": -1}
LOCK_AFTER_STEPS = 10

_seat_box = (np.array([(sa.SHELL_U[0] + sa.SHELL_U[1]) / 2, 0.0, 0.04]),
             np.array([(sa.SHELL_U[1] - sa.SHELL_U[0]) / 2, sa.SHELL_V, 0.066]))
_head = sa.seat_point(sa.HEAD[0][0], 0.0, sa.HEAD[0][1])
_WRIST_CAMERA_MOUNTS = {side: np.asarray(camera.pos) for side, camera in _station.rig().wrist_cams.items()}

_S: dict = {}


def _clear() -> None:
    _S.clear()
    _S.update(head=False, seat=False, latched={"l": False, "r": False}, closed=False, clip_u=sa.CLIP_START_U, tips=None, history=[], depth={"l": 0.0, "r": 0.0}, lock_in={"l": 0, "r": 0}, ready=False,
              fit={}, full_history=[], tongue_positions={}, camera_poses={}, actor_poses={}, clip_actor_poses={}, min_camera_head_gap_m={},
              adjuster_peak=0.0, material_overdraw=False, feed_limit=0.0, mesh_paths={}, t=0)


def _np(x) -> np.ndarray:
    return getattr(x, "torch", x).detach().cpu().numpy().astype(np.float64)


def _rotation(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _fingertips(env) -> np.ndarray:
    base = env.unwrapped
    origin = _np(base.scene.env_origins[0])
    tips = []
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        i = robot.body_names.index("gripper")
        pos = _np(robot.data.body_pos_w)[0, i] - origin
        rot = _rotation(_np(robot.data.body_quat_w)[0, i])
        half_gap = abs(float(scenekit.arm_joints(env, side, ["joint7"])[0]))
        for s in (-1.0, 1.0):
            tips.append(pos + rot @ (TIP_IN_GRIPPER + np.array([s * half_gap, 0.0, 0.0])))
    return np.array(tips)


def _lock(asset, names: list[str]) -> None:
    """Holds these joints where they are now: a latched tongue or a closed clip stays."""
    import torch

    ids = [asset.joint_names.index(n) for n in names]
    now = asset.data.joint_pos
    now = getattr(now, "torch", now)[0, ids].clone().view(1, -1)
    stiff = torch.tensor([[LOCK_STIFFNESS[n.rsplit("_", 1)[0]] for n in names]], device=now.device, dtype=torch.float32)
    asset.set_joint_position_target_index(target=now, joint_ids=ids)
    asset.write_joint_stiffness_to_sim_index(stiffness=stiff, joint_ids=ids)


def _unlock(asset) -> None:
    asset.write_joint_stiffness_to_sim_index(stiffness=0.0 * getattr(asset.data.joint_stiffness, "torch", asset.data.joint_stiffness)[:, :])


def _drive(asset, values: dict[str, float], stiffness: float, damping: float) -> None:
    import torch

    ids = [asset.joint_names.index(name) for name in values]
    target = torch.tensor([list(values.values())], device=asset.device, dtype=torch.float32)
    asset.set_joint_position_target_index(target=target, joint_ids=ids)
    asset.write_joint_stiffness_to_sim_index(stiffness=torch.full_like(target, stiffness), joint_ids=ids)
    asset.write_joint_damping_to_sim_index(damping=torch.full_like(target, damping), joint_ids=ids)


def _set_positions(asset, values: dict[str, float]) -> None:
    import torch

    ids = [asset.joint_names.index(name) for name in values]
    target = torch.tensor([list(values.values())], device=asset.device, dtype=torch.float32)
    asset.write_joint_position_to_sim_index(position=target, joint_ids=ids)
    asset.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(target), joint_ids=ids)


def _constrain_material_feed(asset, tongues, actual_pull) -> None:
    import torch

    available = min(hf.material_feed_limit(side, tongues[tag], sa.material_budget(side)) for tag, side in TAGS.items())
    _S["feed_limit"] = available
    _S["material_overdraw"] |= actual_pull > available + hf.OVERDRAW_TOLERANCE
    limit = torch.tensor([[[0.0, max(available, 1e-6)]]], device=asset.device, dtype=torch.float32)
    asset.write_joint_position_limit_to_sim_index(limits=limit, joint_ids=[asset.joint_names.index("adjuster_pull")])
    _drive(asset, {"adjuster_pull": min(_S["adjuster_peak"], available)}, 3000.0, 0.5)


def _fit_geometry(env, asset, q, body) -> None:
    """Measured material-feed mechanics and independent evidence from rendered ribbon vertices."""
    import omni.usd
    from pxr import UsdGeom, Vt

    started = time.perf_counter()
    stage = omni.usd.get_context().get_stage()
    if not _S["mesh_paths"]:
        wanted = {f"{kind}_{tag}": f"{kind}_{tag}" for tag in TAGS for kind in ("ribbon", "lap")}
        wanted["adjustment_tail"] = "adjustment_tail"
        for prim in stage.Traverse():
            if prim.GetName() in wanted:
                _S["mesh_paths"][wanted[prim.GetName()]] = str(prim.GetPath())
        if len(_S["mesh_paths"]) != 5:
            raise RuntimeError("Shoulder, lap and adjustment-tail ribbon meshes are required")
    index = {name: i for i, name in enumerate(asset.joint_names)}
    actual_tab = body[asset.body_names.index("adjuster")] - np.array([0, 0, TOP])
    actual_pull = max(0.0, sa.ADJUSTER_START[0] - float(actual_tab[0]))
    tail_width = np.array([0.0, hf.STRAP_WIDTH / 2, 0.0])
    tail_points = np.array([np.array(sa.ADJUSTER_EXIT) - tail_width, np.array(sa.ADJUSTER_EXIT) + tail_width,
                            actual_tab - tail_width, actual_tab + tail_width])
    UsdGeom.Mesh.Get(stage, _S["mesh_paths"]["adjustment_tail"]).GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(tail_points.astype(np.float32)))
    _S["adjuster_peak"] = max(_S["adjuster_peak"], actual_pull)
    tongues = {tag: sa.to_seat(body[asset.body_names.index(f"tongue_{tag}")] - np.array([0, 0, TOP]))[0] for tag in TAGS}
    # The cam holds measured feed, with a physical stop when the material
    # needed to reach either actual tongue has exhausted the available budget.
    _constrain_material_feed(asset, tongues, actual_pull)
    _drive(asset, {f"fit_lift_{tag}": 0.0 for tag in TAGS}, 50.0, 0.5)
    fold_targets = {}
    for tag, side in TAGS.items():
        tongue = body[asset.body_names.index(f"tongue_{tag}")] - np.array([0, 0, TOP])
        tongue_local = tongues[tag]
        centres = hf.fitted_centres(side, tongue_local)
        material_floor = sa.material_budget(side) - min(hf.SHOULDER_SLACK, actual_pull) - hf.lap_length(side, tongue_local)
        guide_id = asset.body_names.index(f"fit_body_roll_{tag}")
        guide = sa.to_seat(body[guide_id] - np.array([0, 0, TOP]))[0]
        guide_rot = sa.R_SEAT.T @ _rotation(_np(asset.data.body_quat_w)[0, guide_id])
        route = float(guide[1] - side * 0.035)
        lift = float(guide[2] - (hf.surface_w(0.130, side * 0.035) + 0.001))
        roll = math.atan2(guide_rot[2, 1], guide_rot[1, 1])
        for row, weight in ((1, 1.0), (2, 1.0), (3, 1.0), (4, 0.5), (5, 0.25)):
            centres[row, 1] += route * weight
            centres[row, 2] += lift * weight
        fold_targets[f"fit_fold_{tag}"] = hf.fold_height_for_length(centres, material_floor)
        fold_position = body[asset.body_names.index(f"fit_body_fold_{tag}")] - np.array([0, 0, TOP])
        centres[2] = sa.to_seat(fold_position)[0]
        points = hf.ribbon_points(centres, roll)
        table_points = np.array([sa.X0, 0.0, sa.Z0]) + points.reshape(-1, 3) @ sa.R_SEAT.T
        mesh = UsdGeom.Mesh.Get(stage, _S["mesh_paths"][f"ribbon_{tag}"])
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(table_points.astype(np.float32)))
        mesh.GetDoubleSidedAttr().Set(True)
        faces = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int).reshape(-1, 3)
        normals = np.cross(table_points[faces[:, 1]] - table_points[faces[:, 0]], table_points[faces[:, 2]] - table_points[faces[:, 0]])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(np.repeat(normals, 3, axis=0).astype(np.float32)))
        # Scoring reads back the actual geometry. It does not receive pull,
        # target fold height, or the scripted variant.
        rendered = np.array(mesh.GetPointsAttr().Get(), dtype=float)
        measured = sa.to_seat(rendered).reshape(hf.LONGITUDINAL_SAMPLES, hf.WIDTH_SAMPLES, 3)
        evidence = hf.assess_ribbon(measured, side, tongue_local, material_floor=material_floor)
        _S["fit"][tag] = evidence.record()
        anchor = sa.seat_point(-0.015, side * 0.085, 0.030)
        width = sa.R_SEAT[:, 1] * (hf.STRAP_WIDTH / 2)
        lap = np.array([anchor - width, anchor + width, tongue - width, tongue + width])
        lap_mesh = UsdGeom.Mesh.Get(stage, _S["mesh_paths"][f"lap_{tag}"])
        lap_mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(lap.astype(np.float32)))
        lap_actual = sa.to_seat(np.array(lap_mesh.GetPointsAttr().Get(), dtype=float)).reshape(2, 2, 3)
        _S["fit"][tag]["lap_routed"] = hf.assess_lap(lap_actual, side, tongue_local)
        _S["fit"][tag]["passes"] = evidence.passes and _S["fit"][tag]["lap_routed"]
        _S["fit"][tag]["total_material_m"] = evidence.material_length_m + float(np.linalg.norm(lap_actual[1].mean(axis=0) - lap_actual[0].mean(axis=0)))
        _S["fit"][tag]["fixed_material_budget_m"] = sa.material_budget(side)
        _S["fit"][tag]["guide_pose"] = dict(route_m=route, lift_m=lift, roll_rad=roll,
                                                fold_position_uvw=sa.to_seat(fold_position)[0].tolist())
    _drive(asset, fold_targets, 2500.0, 10.0)
    _S["fit_geometry_ms"] = (time.perf_counter() - started) * 1000.0


def _configure_fixture(asset) -> None:
    """Negative layouts are diagnostic fixtures; the scorer never reads their names."""
    mode = os.environ.get("TASK_DEMO", "success")
    if mode == "twisted":
        _set_positions(asset, {"fit_roll_l": math.pi})
        _drive(asset, {"fit_roll_l": math.pi}, 0.01, 0.02)
    elif mode == "misrouted":
        _set_positions(asset, {"fit_route_l": 0.05})
        _drive(asset, {"fit_route_l": 0.05}, 10.0, 0.2)


def _log_fit() -> None:
    folder = os.environ.get("TASK_SCORE_DIR")
    if not folder:
        return
    import json

    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    row = dict(t=_S["t"] / 20, fit=_S["fit"], latched=_S["latched"], tongue_positions=_S["tongue_positions"], tongue_depth=_S["depth"],
               closed=_S["closed"], clip_u=_S["clip_u"], clip_at_rest=_at_rest(), adjuster_peak=_S["adjuster_peak"],
               head=_S["head"], seat=_S["seat"], material_overdraw=_S["material_overdraw"], feed_limit=_S["feed_limit"],
               tip_to_seat=_S.get("tip_to_seat"), tip_to_adjuster=_S.get("tip_to_adjuster"),
               camera_poses=_S["camera_poses"], min_camera_head_gap_m=_S["min_camera_head_gap_m"],
               actor_poses=_S["actor_poses"], clip_actor_poses=_S["clip_actor_poses"],
               full_condition_window=_S["full_history"], fit_geometry_ms=_S.get("fit_geometry_ms"), score=score({}))
    with (pathlib.Path(folder) / "state.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")


def _record_camera_poses(env) -> None:
    """Diagnostic only: measured gripper poses and the unchanged station camera mounts."""
    base = env.unwrapped
    origin = _np(base.scene.env_origins[0])
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        positions = _np(robot.data.body_pos_w)[0] - origin - np.array([0.0, 0.0, TOP])
        quaternions = _np(robot.data.body_quat_w)[0]
        index = robot.body_names.index("gripper")
        position = positions[index]
        quaternion = quaternions[index]
        _S["actor_poses"][side] = {
            name: dict(position_table_m=positions[robot.body_names.index(name)].tolist(),
                       quat_xyzw=quaternions[robot.body_names.index(name)].tolist())
            for name in ("gripper", "tip_left", "tip_right")
        }
        camera = position + _rotation(quaternion) @ _WRIST_CAMERA_MOUNTS[side]
        gap = float(np.linalg.norm(camera - _head) - sa.HEAD[1])
        _S["camera_poses"][side] = dict(gripper_origin_table_m=position.tolist(), gripper_quat_xyzw=quaternion.tolist(),
                                       optical_origin_table_m=camera.tolist(), head_surface_gap_m=gap)
        _S["min_camera_head_gap_m"][side] = min(_S["min_camera_head_gap_m"].get(side, math.inf), gap)
    harness = base.scene["harness"]
    positions = _np(harness.data.body_pos_w)[0] - origin - np.array([0.0, 0.0, TOP])
    quaternions = _np(harness.data.body_quat_w)[0]
    _S["clip_actor_poses"] = {
        name: dict(position_table_m=positions[harness.body_names.index(name)].tolist(),
                   quat_xyzw=quaternions[harness.body_names.index(name)].tolist())
        for name in ("carriage", "half_l", "half_r")
    }


def _measure(env) -> None:
    base = env.unwrapped
    asset = base.scene["harness"]
    origin = _np(base.scene.env_origins[0])
    names = asset.joint_names
    q = _np(asset.data.joint_pos)[0]
    body = _np(asset.data.body_pos_w)[0] - origin
    tips = _fingertips(env)
    if not _S.get("ready"):
        _clear()
        _unlock(asset)
        _configure_fixture(asset)
        _S["ready"] = True
    _S["t"] += 1
    seat_pose = _np(base.scene["seat"].data.root_link_pose_w)[0]
    tilt = math.degrees(math.acos(float(np.clip(_rotation(seat_pose[3:7])[2, 2], -1.0, 1.0))))
    seat_moved = float(np.linalg.norm(seat_pose[:3] - origin - np.array(SEAT.pos))) > SEAT_MOVED
    _S["seat"] = _S["seat"] or seat_moved or tilt > SEAT_TILT_DEG
    for tag, side in TAGS.items():
        p = body[asset.body_names.index(f"tongue_{tag}")] - np.array([0.0, 0.0, TOP])
        slot = sa.buckle_slot(side)
        tip_z = p[2] - sa.TONGUE[2] / 2 - sa.BLADE[2]
        depth = sa.BUCKLE_TOP - tip_z
        _S["depth"][tag] = float(depth)
        _S["tongue_positions"][tag] = p.tolist()
        aligned = abs(p[0] - slot[0]) < sa.SLOT[0] / 2 and abs(p[1] - slot[1]) < sa.SLOT[1] / 2
        if not _S["latched"][tag] and depth >= LATCH_DEPTH and aligned:
            _S["latched"][tag] = True
            _S["lock_in"][tag] = LOCK_AFTER_STEPS
        if _S["lock_in"][tag] > 0:
            _S["lock_in"][tag] -= 1
            if _S["lock_in"][tag] == 0:  # once the hand has finished pressing it home
                _lock(asset, [f"{k}_{tag}" for k in ("strap_yaw", "strap_pitch", "strap_len", "tongue_pitch", "tongue_yaw")])
    gap = (sa.HALF_GAP + q[names.index("clip_half_l")]) + (sa.HALF_GAP - q[names.index("clip_half_r")])
    if not _S["closed"] and gap <= CLIP_SNAP:
        _S["closed"] = True
        _lock(asset, ["clip_half_l", "clip_half_r"])
    clip_u = sa.CLIP_START_U + float(q[names.index("clip_slide")])
    tips_table = tips - np.array([0.0, 0.0, TOP])
    _S["head"] = _S["head"] or bool(np.any(np.linalg.norm(tips_table - _head, axis=1) - sa.HEAD[1] < NEAR_HEAD))
    local = sa.to_seat(tips_table) - _seat_box[0]
    _S["tip_to_seat"] = float(np.linalg.norm(np.maximum(np.abs(local) - _seat_box[1], 0.0), axis=1).min())
    _S["history"] = (_S["history"] + [clip_u])[-(REST_STEPS + 1):]
    _S.update(clip_u=clip_u, gap=float(gap), tips=tips_table)
    adjuster = body[asset.body_names.index("adjuster")] - np.array([0.0, 0.0, TOP])
    tab_gap = np.maximum(np.abs(tips_table - adjuster) - np.array([0.012, 0.008, 0.005]), 0.0)
    _S["tip_to_adjuster"] = float(np.linalg.norm(tab_gap, axis=1).min())
    _fit_geometry(env, asset, q, body)
    _sample_completion()
    _record_camera_poses(env)
    _log_fit()


@on_reset
def _on_reset(env) -> None:
    _clear()
    _unlock(env.unwrapped.scene["harness"])
    _configure_fixture(env.unwrapped.scene["harness"])
    _measure(env)


@on_step
def _on_step(env) -> None:
    _measure(env)


def _at_rest() -> bool:
    return hf.clip_at_rest(_S["history"], REST_STEPS, REST_MOVE)


def _full_state_ok() -> bool:
    return bool(not _S["head"] and not _S["seat"] and not _S["material_overdraw"]
                and all(_S["latched"].values()) and _S["closed"]
                and abs(_S["clip_u"] - sa.ARMPIT_U) <= ON_LINE
                and len(_S["fit"]) == 2 and all(value["passes"] for value in _S["fit"].values())
                and _S["tip_to_seat"] >= CLEAR and _S["tip_to_adjuster"] >= 0.03)


def _sample_completion() -> None:
    """One observation per control step; score queries never advance time."""
    _S["full_history"] = (_S["full_history"] + [_full_state_ok()])[-(REST_STEPS + 1):]


def score(objects) -> float:
    """0 for a fingertip at the doll's head or a moved seat, at any moment. Otherwise 0.25, 0.5 for the tongues; 0.75 with the clip closed; 1.0
    with the closed clip at armpit level, the grippers clear and all at rest."""
    if not _S or _S["head"] or _S["seat"]:
        return 0.0
    n = sum(_S["latched"].values())
    if n == 0:
        return 0.0
    if n == 1:
        return 0.25
    if not _S["closed"]:
        return 0.5
    if abs(_S["clip_u"] - sa.ARMPIT_U) > ON_LINE:
        return 0.75
    held = len(_S["full_history"]) > REST_STEPS and all(_S["full_history"])
    return 1.0 if _full_state_ok() and held and _at_rest() else 0.75


def success(objects) -> bool:
    steptime.tick()  # the harness calls success() once per control step
    return score(objects) == 1.0


def distance(objects) -> float:
    return abs(_S["clip_u"] - sa.ARMPIT_U) if _S else 1.0


# -- the scripted demo -------------------------------------------------------------------------------------------------

LEFT_CLEAR = (0.18, 0.305, TOP + 0.16)
RIGHT_CLEAR = (0.18, -0.305, TOP + 0.16)
ACROSS = {1: -math.pi / 2, -1: math.pi / 2}  # the grip that closes the fingers along x, per arm
GRIP_ABOVE_BODY = 0.006  # TCP above the middle of the tongue's body, so the pads clear the buckle's top
TRAVEL = 0.165
APPROACH = 0.178
PARTLY_OPEN = -0.022  # wide enough for a tongue; fully open, a finger reaches the clip or the seat's rim


def _tongue_to_buckle(side: int) -> list[Waypoint]:
    yaw = ACROSS[side]
    t = sa.tongue_start(side)
    s = sa.buckle_slot(side)
    if side > 0:
        s = s + np.array([0.0, 0.003, 0.0])
    grip_z = t[2] + GRIP_ABOVE_BODY
    seated = sa.BUCKLE_TOP + sa.TONGUE[2] / 2 + 0.001 + GRIP_ABOVE_BODY
    clear = LEFT_CLEAR if side > 0 else RIGHT_CLEAR
    return [
        Waypoint(1.2, (0.17, side * 0.22, TOP + APPROACH), PARTLY_OPEN, "beside the seat", yaw),
        Waypoint(1.2, (t[0], t[1], TOP + APPROACH), PARTLY_OPEN, "above the tongue", yaw),
        Waypoint(1.0, (t[0], t[1], TOP + grip_z), PARTLY_OPEN, "down to the tongue", yaw),
        Waypoint(0.8, (t[0], t[1], TOP + grip_z), FINGER_CLOSED, "pinch", yaw),
        Waypoint(1.0, (t[0], t[1], TOP + 0.15), FINGER_CLOSED, "lift it out", yaw),
        Waypoint(1.5, (s[0], s[1], TOP + 0.15), FINGER_CLOSED, "over the buckle", yaw),
        Waypoint(1.2, (s[0], s[1], TOP + seated), FINGER_CLOSED, "press it in", yaw),
        Waypoint(0.5, (s[0], s[1], TOP + seated), FINGER_CLOSED, "hold", yaw),
        Waypoint(0.6, (s[0], s[1], TOP + seated), PARTLY_OPEN, "let go", yaw),
        Waypoint(0.8, (s[0], s[1], TOP + 0.145), PARTLY_OPEN, "rise", yaw),
        Waypoint(1.0, (0.15, side * 0.20, TOP + 0.15), PARTLY_OPEN, "back beside the seat", yaw),
        Waypoint(1.5, clear, FINGER_OPEN, "clear"),
    ]


def _clip(to_armpits: bool, watch: float = 3.0) -> list[Waypoint]:
    yaw = 2.8 if to_armpits else 0.0
    start = sa.clip_start()
    finish = sa.seat_point(sa.ARMPIT_U, 0.0, sa.RAIL_W)
    squeeze = FINGER_OPEN * 0.0
    x, z = start[0], start[2]
    moves = [
        Waypoint(2.0 if to_armpits else 1.5, (x, 0.0, TOP + TRAVEL), FINGER_OPEN, "above the clip", yaw),
        Waypoint(1.2, (x, 0.0, TOP + z), FINGER_OPEN, "down to the clip", yaw),
        Waypoint(1.2, (x, 0.0, TOP + z), squeeze, "squeeze it shut", yaw),
    ]
    end_x, end_z = x, z
    if to_armpits:
        moves.append(Waypoint(2.5, (finish[0], 0.0, TOP + finish[2]), squeeze, "slide it up", yaw))
        moves.append(Waypoint(0.4, (finish[0], 0.0, TOP + finish[2]), squeeze, "hold", yaw))
        end_x, end_z = finish[0], finish[2]
    moves.append(Waypoint(0.8, (end_x, 0.0, TOP + end_z), FINGER_OPEN, "let go", yaw))
    if to_armpits:
        # Maintain clip height while separating from the backrest, then
        # keep the hand high until it is outside the seat.
        away = sa.seat_point(sa.ARMPIT_U, -0.080, sa.RAIL_W + 0.045)
        moves += [
            Waypoint(2.0, (away[0], away[1], TOP + away[2]), FINGER_OPEN, "normal out", yaw),
            Waypoint(2.0, (RIGHT_CLEAR[0], -0.250, TOP + away[2]), FINGER_OPEN, "high outside"),
            Waypoint(1.2, RIGHT_CLEAR, FINGER_OPEN, "lower clear"),
        ]
    else:
        moves += [
            Waypoint(1.2, (end_x, 0.0, TOP + 0.175), FINGER_OPEN, "rise", yaw),
            Waypoint(1.5, RIGHT_CLEAR, FINGER_OPEN, "clear"),
        ]
    if watch:
        moves.append(Waypoint(watch, RIGHT_CLEAR, FINGER_OPEN, "watch"))
    return moves


def demo() -> dict[str, list[Waypoint]]:
    """TASK_DEMO=failure leaves the clip at the belly. Open loop, from the starting layout only."""
    mode = os.environ.get("TASK_DEMO", "success")
    to_armpits = mode != "failure"
    left = _tongue_to_buckle(1)
    wait = sum(w.duration for w in left)
    right = [Waypoint(wait, RIGHT_CLEAR, FINGER_OPEN, "wait"), *_tongue_to_buckle(-1), *_clip(False, watch=0.0)]
    if mode not in ("failure", "loose"):
        x, y, z = sa.ADJUSTER_START
        height = TOP + z + 0.003
        right += [
            Waypoint(1.6, (x, y, TOP + 0.12), PARTLY_OPEN, "above the adjustment tail"),
            Waypoint(1.0, (x, y, height), PARTLY_OPEN, "round the adjustment tail"),
            Waypoint(0.8, (x, y, height), FINGER_CLOSED, "grip the adjustment tail"),
            Waypoint(2.0, (x - 0.029, y, height), FINGER_CLOSED, "pull the adjustment tail"),
            Waypoint(0.5, (x - 0.029, y, height), FINGER_CLOSED, "hold the adjustment"),
            Waypoint(0.6, (x - 0.029, y, height), PARTLY_OPEN, "release the adjustment tail"),
            Waypoint(0.8, (x - 0.029, y, TOP + 0.13), PARTLY_OPEN, "rise"),
            Waypoint(2.0, RIGHT_CLEAR, FINGER_OPEN, "clear"),
        ]
    if to_armpits:
        right += _clip(True)
    else:
        right.append(Waypoint(3.0, RIGHT_CLEAR, FINGER_OPEN, "watch"))
    return {"left": left, "right": right}
