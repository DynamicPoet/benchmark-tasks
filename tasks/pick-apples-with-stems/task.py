"""Apple-picking practice station: two 180 g rubber-coated imitation apples on
magnetic breakaway spurs of a free 11.5 kg artificial tree, and a padded picking
bucket. Each apple must come off its spur with its stem, without being squeezed
past the bruise limit or dropped. Sizes and proposed hardware checks are in README.md.
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # lets this file import the helpers beside it

import poseplay  # noqa: E402,F401  (lets demo() hold the gripper level)
import drawplan  # noqa: E402
import scene_assets as sa  # noqa: E402
import scenekit  # noqa: E402
import stemjoint as sj  # noqa: E402
import steptime  # noqa: E402
from poseplay import Joints, Pose  # noqa: E402
from scenekit import UsdBody, on_env_reset, on_reset, on_step, on_substep  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, READY  # noqa: E402
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402
from yam_arena.urdf import axis_angle_matrix  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
PHYSICS_DT = 1.0 / 240.0
GRAVITY = np.array([0.0, 0.0, -9.81])

sa.write_all()

# -- layout (world frame: x forward from the arms, y left, metres) -----------------------------------------------------

TREE_XY = (0.35, -0.10)
BUCKET_XY = (0.20, -0.43)


def _quat(rot: np.ndarray) -> tuple[float, float, float, float]:
    from yam_arena import ik

    q = ik.matrix_to_quat_xyzw(rot)
    return tuple(float(v) for v in q)


def apple_home(k: str) -> tuple[np.ndarray, np.ndarray]:
    """Apple k's centre and rotation in the world as it hangs on the tree."""
    centre, rot = sa.apple_rest(k)
    return centre + np.array([*TREE_XY, TOP]), rot


TREE = UsdBody("tree", usd="assets/tree.usda", pos=(*TREE_XY, TOP + 0.0003), size=(0.30, 0.30, 0.50), mass=sa.TREE_MASS,
               friction=None, solver_iters=16)
APPLES = {k: UsdBody(f"apple_{k}", usd=f"assets/apple_{k}.usda", pos=tuple(float(v) for v in apple_home(k)[0]),
                     rot_xyzw=_quat(apple_home(k)[1]), size=(sa.APPLE_D,) * 3, mass=sa.APPLE_MASS, friction=None,
                     solver_iters=32, torsional_patch=0.004)
          for k in sa.SPURS}
BUCKET = UsdBody("bucket", usd="assets/bucket.usda", pos=(*BUCKET_XY, TOP + 0.0003), size=(0.29, 0.29, 0.23),
                 mass=sa.BKT_MASS, friction=None, solver_iters=16)
OBJECTS = [TREE, *APPLES.values(), BUCKET]
NAMES = {k: f"apple_{k}" for k in sa.SPURS}

# -- how it is scored -----------------------------------------------------------------------------------------------------

DROP_HEIGHT = 0.05      # let go with its lowest point more than this above what it lands on: dropped
OFF_TABLE = 0.10        # an apple whose lowest point is this far below the table top has fallen to the floor
ON_TABLE = 0.005        # an apple outside the bucket with its lowest point this close to the table top is on the table
REST_SPEED = 0.02       # m/s: an apple moving slower than this is at rest
CONTROL_HZ = 20.0
REST_GAP = 0.01         # in the bucket: resting within 1 cm of the pad or of the other apple below it
POT_TIPPED_DEG = 20.0   # the pot leans further than this from upright: tipped over
TOUCH_FORCE = 0.02      # N: a gripper touches an apple
STILL_MOVE = 0.005      # 1.0: neither apple moving more than 5 mm ...
STILL_STEPS = 20        # ... over the last 1 s (21 samples at 20 Hz)
BRUISE_FORCE = 24.0     # N: a finger pad pressing an apple harder than this bruises it (ripe Red Fuji apples' lowest
                        # compression damage force is 24.33 N, doi:10.3390/agriculture16020178)
PADS = (1, 2, 4, 5)     # contact-matrix columns of the four finger pads: left arm tip_left, tip_right; right arm the same
R_APPLE = sa.APPLE_D / 2


def _np(x) -> np.ndarray:
    x = getattr(x, "torch", x)
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x.numpy() if hasattr(x, "numpy") else x, dtype=np.float64)  # warp arrays (contact views)


def _rotation(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def bucket_inner_radius(z: float) -> float:
    """The bucket's inside radius at height z above its base."""
    return sa.BKT_R_BOT - sa.BKT_WALL + (sa.BKT_R_TOP - sa.BKT_R_BOT) * min(max(z, 0.0), sa.BKT_H) / sa.BKT_H


def in_bucket(centre: np.ndarray, others: list[np.ndarray], bucket_pos: np.ndarray, bucket_rot=None) -> bool:
    """The apple rests inside the bucket: its centre inside the wall and below the rim, and sitting on the pad or on the
    other apple (within 1 cm of where it would come to rest), not held up above them."""
    def local_point(p):
        v = p - bucket_pos
        return v if bucket_rot is None else bucket_rot.T @ v

    local = local_point(centre)
    z = float(local[2])
    inside = sa.PAD_TOP < z < sa.BKT_H and float(np.hypot(local[0], local[1])) <= bucket_inner_radius(z) - 0.5 * R_APPLE
    if not inside:
        return False
    if abs(z - (sa.PAD_TOP + R_APPLE)) <= REST_GAP:
        return True
    for other in others:
        lower = local_point(other)
        if abs(lower[2] - (sa.PAD_TOP + R_APPLE)) > REST_GAP or np.linalg.norm(lower[:2]) > bucket_inner_radius(lower[2]):
            continue
        d = np.linalg.norm(centre[:2] - other[:2])
        if d < 2 * R_APPLE and centre[2] > other[2]:
            rest = other[2] + math.sqrt((2 * R_APPLE) ** 2 - d ** 2)
            if abs(centre[2] - rest) <= REST_GAP:
                return True
    return False


def landing_centre_z(centre: np.ndarray, others: list[np.ndarray], bucket_pos: np.ndarray, tree_pos=None) -> float:
    """Where the apple's centre would come to rest if it fell straight down from here: on the bucket's pad, on the other
    apple, on the pot's soil or rim, or on the table."""
    local = centre - bucket_pos
    if float(np.hypot(local[0], local[1])) < bucket_inner_radius(sa.BKT_H):
        z = bucket_pos[2] + sa.PAD_TOP + R_APPLE
    elif float(np.hypot(local[0], local[1])) < sa.BKT_R_TOP + R_APPLE:
        z = bucket_pos[2] + sa.BKT_H + R_APPLE   # on the rim
    else:
        z = TOP + R_APPLE
    tree = np.array([*TREE_XY, TOP]) if tree_pos is None else np.asarray(tree_pos)
    if float(np.hypot(*(centre[:2] - tree[:2]))) < sa.POT_R_TOP + 0.5 * R_APPLE:
        z = max(z, float(tree[2]) + sa.SOIL_Z + R_APPLE)
    for o in others:
        d = float(np.hypot(*(centre[:2] - o[:2])))
        if d < 2 * R_APPLE and o[2] < centre[2]:
            z = max(z, float(o[2]) + math.sqrt(4 * R_APPLE ** 2 - d * d))
    return z


def apple_points(centre: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Six points on the apple's skin and its centre, for the stillness test."""
    return centre + np.vstack([np.zeros(3), R_APPLE * np.vstack([np.eye(3), -np.eye(3)]) @ rot.T])


def sphere_spread(history) -> float:
    """Maximum displacement of any material point of either 75 mm sphere between any two observations."""
    samples = np.asarray(history).reshape(len(history), len(sa.SPURS), 7, 3)
    centres = samples[:, :, 0]
    axes = samples[:, :, 1:4] - centres[:, :, None]
    i, j = np.triu_indices(len(history), 1)
    d = centres[i] - centres[j]
    a = axes[i] - axes[j]
    # A rotation difference maps a sphere onto a disk. Its two equal nonzero
    # singular values give the disk radius, and A*d gives the cross term.
    squared = np.sum(d*d, axis=-1) + 0.5 * np.sum(a*a, axis=(-2, -1))
    squared += 2 * np.linalg.norm(np.einsum("...ij,...j->...i", a, d), axis=-1)
    return float(np.sqrt(np.max(squared)))


def pad_squeeze(forces, axes) -> float:
    """The hardest any finger pad presses the apple: each pad's contact force along its own closing axis (N)."""
    f = np.asarray(forces, float).reshape(-1, 3)
    a = np.asarray(axes, float).reshape(-1, 3)
    if not len(f):
        return 0.0
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    return float(np.max(np.abs(np.sum(f * a, axis=1))))


def track_squeeze(st: dict, squeeze: float) -> None:
    """Latch a bruise: a finger pad ever pressed the apple with more than BRUISE_FORCE."""
    st["peak_squeeze"] = max(st.get("peak_squeeze", 0.0), float(squeeze))
    st["bruised"] = bool(st.get("bruised", False) or squeeze > BRUISE_FORCE)


def track_flight(st, low, touched, landing, dt):
    """Follow a release to its first actual support. Landing is (surface, world contact height), or None."""
    prev = st.get("low")
    was_touched = st.get("touched", False)
    st["low"], st["touched"] = low, touched
    if st["broken"] is None:
        return
    if not touched and (was_touched or "flight_started" not in st):
        st["flight_started"] = True
        st["release_low"] = prev if was_touched and prev is not None else low
        st["drop_from"] = None
    if landing is not None:
        name, surface_z = landing
        if st.get("release_low") is not None:
            st["drop_from"] = st["release_low"] - surface_z
            st["last_landing"] = {"surface": name, "release_low": st["release_low"], "surface_z": surface_z}
            st["dropped"] |= st["drop_from"] > DROP_HEIGHT + 1e-9
            st["release_low"] = None
        if name == "table" and not touched and st.get("peak") is not None:
            st["dropped"] |= st["peak"] - surface_z > DROP_HEIGHT + 1e-9
    if touched:
        st["release_low"] = None
        st["drop_from"] = None
    if touched or st.get("peak") is None or (landing is not None and prev is not None and abs(low-prev) < REST_SPEED*dt):
        st["peak"] = low
    st["peak"] = max(st["peak"], low)
    st["dropped"] |= low < TOP - OFF_TABLE


def track_apple(st: dict, centre: np.ndarray, touched: bool, others: list[np.ndarray], bucket_pos: np.ndarray,
                tree_pos=None) -> None:
    """One control step of an apple's record: dropped (latched), in the bucket, touched. `others`: the other apples that
    are off the tree. Pure, so the scorer can be tested without the simulator.

    Dropped: let go (or knocked off the tree) with its lowest point more than DROP_HEIGHT above what is below it; or,
    while no gripper touches it, it came down onto the table from more than DROP_HEIGHT above it since it was last
    touched or at rest (it rolled off the rim, say); or it reached the floor."""
    # Offline discrete-state callers supply positions only. Infer contact at the
    # current sample; the native path supplies measured contact points to track_flight.
    low = float(centre[2]) - R_APPLE
    z = landing_centre_z(centre, others, bucket_pos, tree_pos) - R_APPLE
    name = "table" if abs(z - TOP) < 1e-6 else "support"
    landing = (name, z) if abs(low-z) <= ON_TABLE else None
    track_flight(st, low, touched, landing, 1 / CONTROL_HZ)
    st["in_bucket"] = st["broken"] is not None and in_bucket(centre, others, bucket_pos)


def apple_value(st: dict) -> float:
    """One apple's share by the state at the end: 0.5, 0.25 or 0. A bruised apple earns nothing."""
    if st["broken"] not in ("seat", "stem") or st["dropped"] or st.get("bruised") or not st["in_bucket"] \
            or st.get("release_low") is not None:
        return 0.0
    return 0.25 if st["broken"] == "stem" else 0.5


# -- what the simulator tells the scorer -----------------------------------------------------------------------------------

_S: dict = {}
_J: dict = {}


def _clear() -> None:
    view = _S.get("view")
    _S.clear()
    _S.update(step=0, tipped=False, history=[], view=view, cache=None, events=[],
              apple={k: dict(broken=None, dropped=False, touched=False, was_touched=False, in_bucket=False, low=0.0,
                             centre=None, rot=None, drop_from=None, bruised=False, peak_squeeze=0.0) for k in sa.SPURS})


_clear()


def _pose(env, name: str) -> tuple[np.ndarray, np.ndarray]:
    base = env.unwrapped
    p = _np(base.scene[name].data.root_link_pose_w)[0]
    return p[:3] - _np(base.scene.env_origins[0]), _rotation(p[3:7])


def _vel(env, name: str) -> tuple[np.ndarray, np.ndarray]:
    v = _np(env.unwrapped.scene[name].data.root_com_vel_w)[0]
    return v[:3], v[3:6]


def _contact_view():
    """Each apple against both grippers, the bucket, tree, table and the other apple."""
    from isaaclab_physx.physics.physx_manager import PhysxManager
    from yam_arena.robot import asset_link_paths

    links = asset_link_paths()
    root = "/World/envs/env_0"
    fingers = [f"{root}/{arm}{links[k]}" for arm in ("LeftArm", "RightArm") for k in ("gripper", "tip_left", "tip_right")]
    sv = PhysxManager.get_physics_sim_view()
    apples = [f"{root}/{NAMES[k]}" for k in sa.SPURS]
    supports = [f"{root}/{name}" for name in ("bucket", "tree", "table")]
    filters = [fingers + supports + [other] for other in reversed(apples)]
    return sv.create_rigid_contact_view(apples, filter_patterns=filters, max_contact_data_count=1024)


def _pad_axes(env) -> np.ndarray:
    """Each finger pad's closing axis in the world, in PADS order (the finger joints slide along the tip's own y)."""
    axes = []
    for side in ("left", "right"):
        robot = env.unwrapped.scene[f"{side}_robot"]
        quats = _np(robot.data.body_quat_w)[0]
        for tip in ("tip_left", "tip_right"):
            axes.append(_rotation(quats[robot.body_names.index(tip)])[:, 1])
    return np.array(axes)


def _contact_sample(env, positions, tree_pose):
    """Read the completed physics interval and track actual landings, including oblique flights."""
    if _S["view"] is None:
        _S["view"] = _contact_view()
    view = _S["view"]
    forces = _np(view.get_contact_force_matrix(dt=PHYSICS_DT)).reshape(len(sa.SPURS), 10, 3)
    fingers = np.linalg.norm(forces[:, :6], axis=2)
    _S["contact_normals_N"] = {k: fingers[i].tolist() for i, k in enumerate(sa.SPURS)}
    axes = _pad_axes(env)
    for i, k in enumerate(sa.SPURS):
        track_squeeze(_S["apple"][k], pad_squeeze(forces[i, list(PADS)], axes))
    # A contact on a wall or a brush against the trunk cannot end a flight.
    # Read detailed PhysX points only when a free apple may have hit a support.
    detail = None
    bucket_pose = None
    for i, k in enumerate(sa.SPURS):
        st = _S["apple"][k]
        touched = bool(np.max(fingers[i]) > TOUCH_FORCE)
        centre = positions[k]
        landing = None
        candidates = np.flatnonzero(np.abs(forces[i, 6:, 2]) > TOUCH_FORCE)
        if st["broken"] is not None and candidates.size and (not touched or st.get("release_low") is not None):
            if detail is None:
                detail = tuple(_np(v) for v in view.get_contact_data(dt=PHYSICS_DT))
                origin = _np(env.unwrapped.scene.env_origins[0])
            impulse, points, normals, _, counts, starts = detail
            for col in candidates:
                name = ("bucket", "tree", "table", "apple")[col]
                if name == "apple" and not all(j.broken for key, j in _J.items() if key in sa.SPURS and key != k):
                    continue
                start = int(starts[i, 6+col])
                for index in range(start, start + int(counts[i, 6+col])):
                    point = points[index] - origin
                    if abs(impulse[index, 0]) <= TOUCH_FORCE or point[2] >= centre[2] or abs(normals[index, 2]) < 0.10:
                        continue
                    if name in ("bucket", "tree"):
                        if name == "bucket" and bucket_pose is None:
                            bucket_pose = _pose(env, "bucket")
                        pos, rot = bucket_pose if name == "bucket" else tree_pose
                        local = rot.T @ (point - pos)
                        heights = (sa.PAD_TOP, sa.BKT_H) if name == "bucket" else (sa.SOIL_Z, sa.POT_H)
                        if min(abs(local[2] - z) for z in heights) > ON_TABLE:
                            continue
                    if landing is None or point[2] > landing[1]:
                        landing = (name, float(point[2]))
        track_flight(st, float(centre[2]) - R_APPLE, touched, landing, PHYSICS_DT)


def _show(path: str, visible: bool) -> None:
    import isaaclab.sim as sim_utils
    from pxr import UsdGeom

    prim = sim_utils.get_current_stage().GetPrimAtPath(f"/World/envs/env_0/{path}")
    if prim.IsValid():
        img = UsdGeom.Imageable(prim)
        img.MakeVisible() if visible else img.MakeInvisible()


def _show_state(k: str, broken: str | None) -> None:
    """The tree and the apple as they look after `broken`: after a stem pull the stem stays on the spur and the apple
    shows its empty socket."""
    pulled = broken == "stem"
    for part in ("stem", "cap"):
        _show(f"tree/visual/{part}_{k}", pulled)
        _show(f"{NAMES[k]}/visual/{part}", not pulled)
    for part in ("socket", "socket_magnet"):
        _show(f"{NAMES[k]}/visual/{part}", True)


def _new_joints(env) -> None:
    _J.clear()
    tree_rot = np.eye(3)
    for k in sa.SPURS:
        centre, rot = sa.apple_rest(k)
        j = sj.Joint(name=k, anchor=sa.ABSCISSION.copy(), point=sa.abscission_in_tree(k), rot0=rot.copy(),
                     mass=sa.APPLE_MASS, inertia=0.4 * sa.APPLE_MASS * R_APPLE ** 2)
        j.settle(GRAVITY, tree_rot)
        _J[k] = j
        if env is not None:
            _show_state(k, None)
    _J["_zeroed"] = set()


@on_env_reset
def _env_reset(env) -> None:
    """A reset puts both apples back on their spurs, whole."""
    _new_joints(env)
    _clear()
    _finger_limits(env)


@on_reset
def _reset(env) -> None:
    """Failures stay failed until a real reset (the settling steps after it do not count)."""
    _clear()
    _finger_limits(env)
    for k in sa.SPURS:
        if _J[k].broken is not None:  # cannot happen while settling; if it did, the trial starts with it
            _S["apple"][k]["broken"] = _J[k].broken


def _finger_limits(env) -> None:
    """Set and read the practice station's 40 N per-finger effort envelope."""
    import torch

    limits = {}
    for side in ("left", "right"):
        robot = env.unwrapped.scene[f"{side}_robot"]
        ids = [robot.joint_names.index(n) for n in ("joint7", "joint8")]
        robot.write_joint_effort_limit_to_sim_index(
            limits=torch.full((1, 2), 40., device=robot.device), joint_ids=ids)
        actual = _np(robot.root_physx_view.get_dof_max_forces())[0, ids]
        if not np.allclose(actual, 40., atol=1e-4, rtol=0):
            raise RuntimeError("The native per-finger effort cap is not 40 N")
        limits[side] = actual.tolist()
    _S["gripper_limits"] = limits


def _set_wrench(env, name: str, force: np.ndarray, torque: np.ndarray) -> None:
    import torch

    asset = env.unwrapped.scene[name]
    dev = env.unwrapped.device
    asset.permanent_wrench_composer.set_forces_and_torques_index(
        forces=torch.tensor(force, device=dev, dtype=torch.float32).view(1, 1, 3),
        torques=torch.tensor(torque, device=dev, dtype=torch.float32).view(1, 1, 3), is_global=True)


@on_substep
def _substep(env) -> None:
    """Every physics step: each apple still on the tree is held by its joint spring; a joint loaded past a limit lets go."""
    if not _J:
        _new_joints(env)
    tree_pos, tree_rot = _pose(env, "tree")
    if _tilt_deg(tree_rot) > POT_TIPPED_DEG:
        _S["tipped"] = True
    tree_v, tree_w = _vel(env, "tree")
    tree_com = tree_pos + tree_rot @ np.array([0.0, 0.0, sa.TREE_COM_Z])
    total_f, total_t = np.zeros(3), np.zeros(3)
    positions = {}
    released = False
    for k in sa.SPURS:
        j: sj.Joint = _J[k]
        pos, rot = _pose(env, NAMES[k])
        positions[k] = pos
        if j.broken is not None:
            if k not in _J["_zeroed"]:
                _set_wrench(env, NAMES[k], np.zeros(3), np.zeros(3))
                _J["_zeroed"].add(k)
            continue
        v, w = _vel(env, NAMES[k])
        force, moment, p_tree, loads = j.wrench(pos, rot, v, w, tree_pos, tree_rot, tree_com, tree_v, tree_w, PHYSICS_DT)
        loads["uproll_deg"] = float(np.degrees(np.arctan2(-rot[2, 0], np.hypot(rot[0, 0], rot[1, 0]))))
        part = j.check(loads)
        if part is not None:
            j.broken = part
            released = True
            _S["apple"][k]["broken"] = part
            _show_state(k, part)
            _S["events"].append((round(_S["step"] / 20.0, 3), k, part, {n: round(x, 3) for n, x in loads.items()}))
            _set_wrench(env, NAMES[k], np.zeros(3), np.zeros(3))
            _J["_zeroed"].add(k)
            continue
        _set_wrench(env, NAMES[k], force, moment + np.cross(rot @ j.anchor, force))
        total_f -= force
        total_t += np.cross(p_tree - tree_com, -force) - moment
    _set_wrench(env, "tree", total_f, total_t)
    # The final interval of a control has already been read by _step. A new
    # break still needs the contact state at its own release instant.
    previous_read = _S.pop("control_contact_read", False)
    if released or (not previous_read and any(_J[k].broken is not None for k in sa.SPURS)):
        _contact_sample(env, positions, (tree_pos, tree_rot))


def _tilt_deg(rot: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(rot[2, 2], -1.0, 1.0))))


@on_step
def _step(env) -> None:
    _S["step"] += 1
    tree_pos, tree_rot = _pose(env, "tree")
    if _tilt_deg(tree_rot) > POT_TIPPED_DEG:
        _S["tipped"] = True
    bucket_pos, bucket_rot = _pose(env, "bucket")
    poses = {k: _pose(env, NAMES[k]) for k in sa.SPURS}
    _contact_sample(env, {k: p[0] for k, p in poses.items()}, (tree_pos, tree_rot))
    _S["control_contact_read"] = True
    for k in sa.SPURS:
        st = _S["apple"][k]
        j: sj.Joint = _J[k]
        centre, rot = poses[k]
        st["centre"], st["rot"] = centre, rot
        if j.broken is not None and st["broken"] is None:
            st["broken"] = j.broken
            _show_state(k, j.broken)
        others = [poses[o][0] for o in sa.SPURS if o != k and _S["apple"][o]["broken"] is not None]
        st["in_bucket"] = j.broken is not None and in_bucket(centre, others, bucket_pos, bucket_rot)
    _S["history"].append(np.vstack([apple_points(*poses[k]) for k in sa.SPURS]))
    del _S["history"][:-(STILL_STEPS + 1)]
    if os.environ.get("TASK_DEBUG") and os.environ.get("TASK_SCORE_DIR"):
        _log(env, poses)


def _log(env, poses) -> None:
    """Debug telemetry (TASK_DEBUG=1), every control step: apple poses, joint peaks, touches, the right TCP."""
    import json

    from yam_arena.constants import TCP_OFFSET

    base = env.unwrapped
    robot = base.scene["right_robot"]
    i = robot.body_names.index("gripper")
    g = _np(robot.data.body_pos_w)[0, i] - _np(base.scene.env_origins[0])
    grot = _rotation(_np(robot.data.body_quat_w)[0, i])
    tcp = g + grot @ np.array(TCP_OFFSET)
    row = {"t": _S["step"] / 20.0, "tcp_r": np.round(tcp, 4).tolist(), "tool_x": np.round(grot[:, 0], 3).tolist(),
           "tool_z": np.round(grot[:, 2], 3).tolist(),
           "q_r": np.round(scenekit.arm_joints(env, "right", [f"joint{n}" for n in range(1, 9)]), 4).tolist(),
           "tree": np.round(np.r_[_pose(env, "tree")[0], _tilt_deg(_pose(env, "tree")[1])], 4).tolist(),
           "events": _S["events"], "native_gripper_effort_limits_N": _S["gripper_limits"]}
    for k in sa.SPURS:
        st = _S["apple"][k]
        c, r = poses[k]
        row[k] = {"c": np.round(c, 4).tolist(), "x": np.round(r[:, 0], 3).tolist(), "broken": st["broken"],
                  "touch": bool(st["touched"]), "dropped": bool(st["dropped"]), "in": bool(st["in_bucket"]),
                  "bruised": bool(st["bruised"]), "peak_squeeze_N": round(float(st["peak_squeeze"]), 3),
                  "drop_from": None if st["drop_from"] is None else round(float(st["drop_from"]), 4),
                  "last_landing": st.get("last_landing"),
                  "peak": {n: round(x, 3) for n, x in _J[k].peak.items()},
                  "actual_uproll_deg":round(math.degrees(math.atan2(-r[2,0],math.hypot(r[0,0],r[1,0]))),3),
                  "contact_normal_N_by_left_gripper_tips_right_gripper_tips":_S.get("contact_normals_N",{}).get(k)}
        if st["broken"] is None and st["centre"] is not None:
            row[k]["rel_tcp"] = np.round(grot.T @ (c - tcp), 4).tolist()
    folder = pathlib.Path(os.environ["TASK_SCORE_DIR"])
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "telemetry.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def _evaluate() -> dict:
    if _S["cache"] is not None and _S["cache"][0] == _S["step"]:
        return _S["cache"][1]
    values = {k: apple_value(_S["apple"][k]) for k in sa.SPURS}
    total = 0.0 if _S["tipped"] else sum(values.values())
    hist = _S["history"]
    if len(hist) >= STILL_STEPS + 1:
        spread = sphere_spread(hist)
    else:
        spread = math.inf
    touched = any(_S["apple"][k]["touched"] for k in sa.SPURS)
    out = dict(values=values, total=total, spread=spread, still=spread <= STILL_MOVE, touched=touched)
    _S["cache"] = (_S["step"], out)
    return out


def score(objects) -> float:
    """0 if the pot ever tips over. Otherwise each apple earns 0.5 (in the bucket, stem on, neither dropped nor bruised)
    or 0.25 (the same with its stem pulled out), and the sum is capped at 0.75: 1.0 comes only from success(), held 1 s."""
    return min(_evaluate()["total"], 0.75)


def success(objects) -> bool:
    """Both apples earn 0.5, no gripper touches either, and neither moves more than 5 mm over the last 1 s. The harness
    then needs it for 1 s in a row."""
    steptime.tick()  # the harness calls success() once per control step
    e = _evaluate()
    if os.environ.get("TASK_DEBUG") and _S["step"] % 20 == 0:
        a = _S["apple"]
        print(f"[task] t={_S['step'] / 20:.1f} total={e['total']} " + " ".join(
            f"{k}:{a[k]['broken']}/drop={a[k]['dropped']}/in={a[k]['in_bucket']}/touch={a[k]['touched']}"
            f"/squeeze={a[k]['peak_squeeze']:.1f}" for k in sa.SPURS)
            + f" spread={e['spread']:.4f} tipped={_S['tipped']} events={_S['events']}", flush=True)
    return e["total"] == 1.0 and not e["touched"] and e["still"]


# -- the scripted demo -------------------------------------------------------------------------------------------------------
# The right arm picks both apples, upper first. It aims from where the cameras show each apple, and only changes its own
# targets; nothing in the simulation is moved or set.


def _rz(a: float) -> np.ndarray:
    return axis_angle_matrix(np.array([0.0, 0.0, 1.0]), a)


def _ry(a: float) -> np.ndarray:
    return axis_angle_matrix(np.array([0.0, 1.0, 0.0]), a)


R_LEVEL = _ry(-math.pi / 2) @ grip_rotation(0.0)   # the gripper pointing along +x, level, fingers closing along y


def level_grip(approach: np.ndarray, twist: float = 0.0, lift: float = 0.0) -> np.ndarray:
    """The gripper pointing along `approach` (level), turned `twist` about it and tilted up by `lift` (rad)."""
    a = np.asarray(approach, float)
    yaw = math.atan2(a[1], a[0])
    side = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    return axis_angle_matrix(side, -lift) @ _rz(yaw) @ R_LEVEL @ axis_angle_matrix(np.array([0.0, 0.0, 1.0]), twist)


def approach_dir(k: str) -> np.ndarray:
    _, d = sa.spur_frame(k)
    return -d


UPROLL = math.radians(60.0)
GRASP_CLOCK = {"a": -math.pi/2, "b": math.pi/2}  # fingers above/below the fruit; front-facing palm
HOLD_FINGER = -0.0330           # 3.6 mm inside the observed 36.6 mm half-opening: about 18 N per finger at 5000 N/m
SQUEEZE_FINGER = FINGER_CLOSED  # the squeeze counterexample: the finger drive saturates at the native 40 N cap
PULL = 0.10                     # TASK_DEMO=straight: the hand pulls this far straight out along the apple's stem
PLACE_GAP = 0.010               # the apple is let go with its lowest point this far above the pad
SLOTS = {"b": (-0.03, -0.042), "a": (-0.03, 0.042)}   # where each apple goes in the bucket, from its middle (x, y)
DOWN_YAW = -math.pi / 2         # fingers close along x; this wrist branch remains reachable through the bucket
CARRY_Z = TOP + 0.30            # the TCP's height over the bucket before lowering (the apple's bottom 3 cm over the rim)
MID_Z = TOP + 0.19              # ... and part way down, where the hand has turned almost straight down


def _bucket_slot(k: str) -> np.ndarray:
    env = scenekit.env()
    centre = _pose(env, "bucket")[0] if env is not None else np.array([*BUCKET_XY, TOP])
    dx, dy = SLOTS[k]
    return np.array([centre[0] + dx, centre[1] + dy, centre[2] + sa.PAD_TOP + PLACE_GAP + R_APPLE])


def _seen_apple(k: str):
    """The apple's centre as the cameras show it, read when the waypoint starts (offline: where it hangs)."""
    def target(plan):
        env = scenekit.env()
        return _pose(env, NAMES[k])[0] if env is not None else apple_home(k)[0]
    return target


def _from_apple(k: str, back: float, up: float = 0.0):
    def target(plan):
        return np.asarray(_seen_apple(k)(plan)) - back * approach_dir(k) + np.array([0.0, 0.0, up])
    return target


def _along_stem(k: str, dist: float, up_deg: float = 0.0):
    """The apple's centre moved `dist` straight out along its own stem, away from the seat, aimed `up_deg` above the
    stem, read when the move begins (the apple's x axis points along the stem toward the seat)."""
    def target(plan):
        env = scenekit.env()
        centre, rot = _pose(env, NAMES[k]) if env is not None else apple_home(k)
        u = -rot[:, 0]
        n = np.array([0.0, 0.0, 1.0]) - u[2] * u
        n /= np.linalg.norm(n)
        a = math.radians(up_deg)
        return np.asarray(centre) + dist * (math.cos(a) * u + math.sin(a) * n)
    return target


def _over_slot(k: str, z: float | None = None):
    def target(plan):
        p = _bucket_slot(k)
        return np.array([p[0], p[1], CARRY_Z if z is None else z])
    return target


def _slot(k: str):
    return lambda plan: _bucket_slot(k)


def _uproll_targets(k: str, grip: np.ndarray):
    """Cache the observed seat and apple once, then make an actual upward arc.

    The hand may approach from world+X independently of the spur azimuth. The
    fruit still rolls about the horizontal axis across its OWN stem, not yaw.
    """
    cache={}
    def frame():
        if not cache:
            env=scenekit.env()
            c=_pose(env,NAMES[k])[0]if env is not None else apple_home(k)[0]
            tp,tr=_pose(env,"tree")if env is not None else(np.array([*TREE_XY,TOP]),np.eye(3))
            p=tp+tr@sa.abscission_in_tree(k);v=c-p;axis=np.cross(v,[0.,0.,1.]);axis/=np.linalg.norm(axis)
            cache.update(p=p,v=v,axis=axis)
        return cache
    def position(angle,draw=0.):
        def target(plan):
            f=frame();A=axis_angle_matrix(f['axis'],angle);v=A@f['v']
            return f['p']+v+draw*v/np.linalg.norm(v)
        return target
    def rotation(angle):
        return lambda plan:axis_angle_matrix(frame()['axis'],angle)@grip
    return position,rotation


_SEEDS = [np.array(READY, float), np.array([0.6, 1.6, 1.2, 0.4, 0.0, 0.0]), np.array([0.2, 1.9, 1.9, 0.0, -1.0, 0.0]),
          np.array([-0.8, 1.4, 1.5, -1.0, -0.4, 0.5]), np.array([-1.1, 1.5, 1.6, -1.4, -0.4, 0.5])]


def _joints_for(pos_fn, rot: np.ndarray):
    """A Joints target: the arm's joints that put the TCP at pos_fn(plan) with rotation rot, the IK started from the
    plan's current joints and a few fixed seeds; of the solutions, the one nearest the current joints."""
    from yam_arena import ik
    from yam_arena.station import BASES

    def q(plan):
        pos = np.asarray(pos_fn(plan) if callable(pos_fn) else pos_fn, float)
        best = None
        for seed in [plan.q.copy()] + _SEEDS:
            sol, pe, re = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), seed, base_pos=BASES["right"], iters=300)
            cost = (pe > 0.002 or re > 0.02, float(np.abs(sol - plan.q).max()))
            if best is None or cost < best[0]:
                best = (cost, sol)
        return tuple(float(v) for v in best[1])
    return q


def _axial_reference(k, grip, up_rotation, cache):
    """Read the real drive contract, then return stock arm POSITION targets only."""
    from yam_arena import ik
    from yam_arena.station import BASES

    def target(plan):
        env = scenekit.env()
        if env is None:
            if os.environ.get("TASK_SCORE_DIR"):
                raise RuntimeError("A native draw cannot use the offline gain reference")
            actual = plan.q.copy()
            axis = up_rotation(plan) @ grip.T @ apple_home(k)[1][:, 0]
            gains = np.array([80., 80., 80., 10., 10., 10.])
            bounds = np.c_[ik.LOWER, ik.UPPER]
            efforts = np.array([27., 27., 27., 7., 7., 7.])
            velocities = np.array([5.5, 5.5, 5.5, 20., 20., 20.])
            contract = {"scope": "offline configured reference; native contract still required"}
        else:
            robot = env.unwrapped.scene["right_robot"]
            ids = [robot.joint_names.index(f"joint{i}") for i in range(1, 7)]
            view = robot.root_physx_view
            actual = _np(view.get_dof_positions())[0, ids].copy()
            gains = _np(view.get_dof_stiffnesses())[0, ids].copy()
            damping = _np(view.get_dof_dampings())[0, ids].copy()
            bounds = _np(view.get_dof_limits())[0, ids].copy()
            efforts = _np(view.get_dof_max_forces())[0, ids].copy()
            native_velocity = _np(view.get_dof_max_velocities())[0, ids].copy()
            drive_type = _np(view.get_drive_types())[0, ids].copy()
            after_reads = _np(view.get_dof_positions())[0, ids].copy()
            contract = dict(scope="native read-only contract", joint_names=[robot.joint_names[i] for i in ids],
                            drive_type=drive_type.tolist(), stiffness=gains.tolist(), damping=damping.tolist(),
                            limits=bounds.tolist(), efforts=efforts.tolist(), native_velocity=native_velocity.tolist(),
                            q_before=actual.tolist(), q_after=after_reads.tolist(),
                            getters_left_q_unchanged=bool(np.array_equal(actual, after_reads)))
            if os.environ.get("TASK_SCORE_DIR"):
                import json
                with open(pathlib.Path(os.environ["TASK_SCORE_DIR"]) / "draw-contract.jsonl", "a") as f:
                    f.write(json.dumps(dict(apple=k, t=_S["step"] / 20., **contract)) + "\n")
            if not np.array_equal(actual, after_reads):
                raise RuntimeError("The native draw contract changed joint state while being read")
            if not np.all(drive_type == 1) or not np.allclose(gains, [80, 80, 80, 10, 10, 10], atol=1e-4, rtol=0):
                raise RuntimeError("The axial planner requires the unchanged native SI force-drive gains")
            if not np.allclose(bounds, np.c_[ik.LOWER, ik.UPPER], atol=1e-4, rtol=0):
                raise RuntimeError("Native and physical robot joint limits disagree")
            velocities = np.minimum(native_velocity, [5.5, 5.5, 5.5, 20, 20, 20])
            axis = _pose(env, NAMES[k])[1][:, 0]
        q, report = drawplan.reference(plan.q.copy(), actual, axis, gains, bounds, efforts, velocities,
                                       BASES["right"], force_N=22., ramp_s=4.)
        cache["q_target"] = q.copy()
        if env is not None and os.environ.get("TASK_SCORE_DIR"):
            import json
            report.update(apple=k, t=_S["step"] / 20., contract=contract, stock_joint_position_actions_only=True)
            with open(pathlib.Path(os.environ["TASK_SCORE_DIR"]) / "draw-reference.jsonl", "a") as f:
                f.write(json.dumps(report) + "\n")
        return tuple(float(v) for v in q)
    return target


def pick(k: str, pull: bool, last: bool = False, grip: float = HOLD_FINGER) -> list:
    """Upward roll and draw; with pull=True, a straight pull along the stem without rolling."""
    a = np.array([1.,0.,0.])
    o, cl = FINGER_OPEN, grip
    level = level_grip(a,GRASP_CLOCK[k])
    down = grip_rotation(DOWN_YAW)
    tilted = _ry(-math.radians(45.0)) @ down    # pointing forward and down: the arm cannot hold it straight down this high
    steep = _ry(-math.radians(12.0)) @ down
    zero = (0.0, 0.0, 0.0)
    seen=_seen_apple(k)
    front=lambda back:(lambda plan:np.asarray(seen(plan))-back*a)
    w = [
        Joints(4.0, zero, o, f"in front of apple {k}", q=_joints_for(front(.10), level)),
        Pose(2.0, front(0.), o, f"round apple {k}", rot=level),
        Pose(1.2, front(0.), cl, f"close on apple {k}", rot=level),
        Pose(0.8, front(0.), cl, f"hold apple {k}", rot=level),
    ]
    if pull:
        w += [Pose(3.0, _along_stem(k, PULL), cl, f"pull apple {k} straight off along its stem", rot=level)]
    else:
        position,rotation=_uproll_targets(k,level)
        for index in range(1,25):
            angle=UPROLL*index/24
            w.append(Pose(.2,position(angle),cl,f"roll apple {k} UP {index*2.5:g}deg",rot=rotation(angle)))
        w += [Pose(1.,position(UPROLL),cl,f"settle apple {k} at upward roll",rot=rotation(UPROLL)),
              Pose(4.,position(UPROLL,.002),cl,f"draw apple {k} from its seat",rot=rotation(UPROLL))]
    if not pull:  # the straight-pull counterexample is already 10 cm clear
        # A bounded joint-position bias plans an axial load while retaining the
        # UP baseline. It never sends an effort or force action to the robot.
        draw_cache = {}
        w += [Joints(4., zero, cl, f"load apple {k} axially from the UP baseline",
                     q=_axial_reference(k, level, rotation(UPROLL), draw_cache))]
        def released_centre(plan):
            if scenekit.env() is not None:
                if _J[k].broken is None:
                    raise RuntimeError(f"Apple {k} did not physically release during the bounded draw; no transport")
                return np.asarray(seen(plan))
            from yam_arena import ik
            from yam_arena.station import BASES
            return ik.tcp_pose(draw_cache["q_target"], BASES["right"])[0]
        def released_hold(plan):
            released_centre(plan)  # physical release is required before free handling
            return tuple(float(v) for v in draw_cache["q_target"])
        w += [Joints(1.,zero,cl,f"settle released apple {k}",q=released_hold),
              Pose(2.,released_centre,cl,f"unroll the released apple {k}",rot=level)]
    w += [
        Pose(3.,(.20,-.30,TOP+.34),cl,f"apple {k} along the clear front corridor",rot=level),
        Joints(4.0, zero, cl, f"apple {k} over the bucket", q=_joints_for(_over_slot(k), tilted)),
        Pose(1., _over_slot(k, TOP+.26), cl, f"apple {k} above the rim", rot=_ry(-math.radians(30))@down),
        Pose(1., _over_slot(k, TOP+.23), cl, f"apple {k} through the rim", rot=_ry(-math.radians(15))@down),
        Pose(1., _over_slot(k, MID_Z), cl, f"apple {k} into the bucket", rot=down),
        Pose(1.5, _slot(k), cl, f"lower apple {k} to the pad", rot=down),
        Pose(0.8, _slot(k), o, f"let go of apple {k}", rot=down),
    ]
    if last:  # out of the camera's way quickly: the trial ends 2 s after both apples are still
        w += [Pose(1., _over_slot(k, MID_Z), o, f"up from apple {k}", rot=down),
              Pose(1., _over_slot(k, TOP+.23), o, f"clear the rim ({k})", rot=_ry(-math.radians(15))@down),
              Pose(1., _over_slot(k), o, f"out of the bucket ({k})", rot=tilted),
              Joints(2., zero, o, "back to the start", q=tuple(READY))]
    else:
        w += [Pose(1.2, _over_slot(k, MID_Z), o, f"up from apple {k}", rot=down),
              Pose(1., _over_slot(k, TOP+.23), o, f"clear the rim ({k})", rot=_ry(-math.radians(15))@down),
              Pose(1.2, _over_slot(k), o, f"out of the bucket ({k})", rot=tilted),
              Joints(2.,zero,o,"clear before the next apple",q=tuple(READY))]
    return w


def demo():
    """The right arm picks both apples, the upper one first, and lays each on the bucket's pad. TASK_DEMO=straight pulls
    each apple straight out along its stem instead of rolling it, then lays it in the same way. TASK_DEMO=squeeze picks
    as normal but holds the first apple with the fingers fully closed (40 N each): that apple is bruised."""
    mode = os.environ.get("TASK_DEMO")
    pull = mode == "straight"
    first = SQUEEZE_FINGER if mode == "squeeze" else HOLD_FINGER
    w = pick("b", pull, grip=first) + pick("a", pull, last=True)
    w += [Joints(4.0, (0.0, 0.0, 0.0), FINGER_OPEN, "wait", q=tuple(READY))]
    return {"right": w}
