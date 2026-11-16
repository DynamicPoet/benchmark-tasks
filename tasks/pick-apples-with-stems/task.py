"""On the station table: a columnar apple tree in a black plastic pot (300 mm across, 220 mm tall, filled with potting mix)
stands in front of the left arm. Its single trunk rises to 500 mm above the table. On the trunk's side towards the right
arm, two short fruiting spurs carry one ripe red 'Gala' apple each (75 mm across, 180 g), on level stems 20 mm long: the
lower apple's centre is 300 mm above the table, the upper one's 380 mm. Between the arms, in front, stands a green plastic
picking bucket (290 mm across, 230 mm deep) with a grey foam pad on its floor.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \\
           -E cam_width=320 -E cam_height=176 --scorer task_score --max-steps 2400 --instruction "<the task.json instruction>"
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # lets this file import the helpers beside it

import poseplay  # noqa: E402,F401  (lets demo() hold the gripper level)
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

TREE_XY = (0.42, 0.10)
BUCKET_XY = (0.22, -0.14)


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


def in_bucket(centre: np.ndarray, others: list[np.ndarray], bucket_pos: np.ndarray) -> bool:
    """The apple rests inside the bucket: its centre inside the wall and below the rim, and sitting on the pad or on the
    other apple (within 1 cm of where it would come to rest), not held up above them."""
    local = centre - bucket_pos
    z = float(local[2])
    inside = sa.PAD_TOP < z < sa.BKT_H and float(np.hypot(local[0], local[1])) <= bucket_inner_radius(z) - 0.5 * R_APPLE
    return bool(inside and float(centre[2]) - landing_centre_z(centre, others, bucket_pos, None) < REST_GAP)


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


def track_apple(st: dict, centre: np.ndarray, touched: bool, others: list[np.ndarray], bucket_pos: np.ndarray,
                tree_pos=None) -> None:
    """One control step of an apple's record: dropped (latched), in the bucket, touched. `others`: the other apples that
    are off the tree. Pure, so the scorer can be tested without the simulator.

    Dropped: let go (or knocked off the tree) with its lowest point more than DROP_HEIGHT above what is below it; or,
    while no gripper touches it, it came down onto the table from more than DROP_HEIGHT above it since it was last
    touched or at rest (it rolled off the rim, say); or it reached the floor."""
    prev = st.get("prev")
    speed = math.inf if prev is None else float(np.linalg.norm(centre - prev)) * CONTROL_HZ
    st["prev"] = np.asarray(centre, float).copy()
    st["touched"] = touched
    low = float(centre[2]) - R_APPLE
    st["low"] = low
    if st["broken"] is None:
        return
    if not touched and st["drop_from"] is None:
        st["drop_from"] = float(centre[2]) - landing_centre_z(centre, others, bucket_pos, tree_pos)
        if st["drop_from"] > DROP_HEIGHT:
            st["dropped"] = True
    if touched:
        st["drop_from"] = None
    if touched or speed < REST_SPEED or st.get("peak") is None:
        st["peak"] = low
    st["peak"] = max(st["peak"], low)
    on_table = low < TOP + ON_TABLE and float(np.hypot(*(centre[:2] - bucket_pos[:2]))) > sa.BKT_R_TOP
    if not touched and on_table and st["peak"] - low > DROP_HEIGHT:
        st["dropped"] = True
    if low < TOP - OFF_TABLE:
        st["dropped"] = True
    st["in_bucket"] = in_bucket(centre, others, bucket_pos)


def apple_value(st: dict) -> float:
    """One apple's share by the state at the end: 0.5, 0.25 or 0."""
    if st["broken"] is None or st["dropped"] or not st["in_bucket"] or st["broken"] == "spur":
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
                             centre=None, rot=None, drop_from=None) for k in sa.SPURS})


_clear()


def _pose(env, name: str) -> tuple[np.ndarray, np.ndarray]:
    base = env.unwrapped
    p = _np(base.scene[name].data.root_link_pose_w)[0]
    return p[:3] - _np(base.scene.env_origins[0]), _rotation(p[3:7])


def _vel(env, name: str) -> tuple[np.ndarray, np.ndarray]:
    v = _np(env.unwrapped.scene[name].data.root_com_vel_w)[0]
    return v[:3], v[3:6]


def _contact_view():
    """PhysX contact view: each apple against both grippers and their fingers."""
    from isaaclab_physx.physics.physx_manager import PhysxManager
    from yam_arena.robot import asset_link_paths

    links = asset_link_paths()
    root = "/World/envs/env_0"
    fingers = [f"{root}/{arm}{links[k]}" for arm in ("LeftArm", "RightArm") for k in ("gripper", "tip_left", "tip_right")]
    sv = PhysxManager.get_physics_sim_view()
    return sv.create_rigid_contact_view([f"{root}/{NAMES[k]}" for k in sa.SPURS], filter_patterns=[fingers] * len(sa.SPURS),
                                        max_contact_data_count=512)


def _touching() -> dict:
    if _S["view"] is None:
        _S["view"] = _contact_view()
    f = _np(_S["view"].get_contact_force_matrix(dt=PHYSICS_DT)).reshape(len(sa.SPURS), -1, 3)
    return {k: bool(np.linalg.norm(f[i], axis=1).max() > TOUCH_FORCE) for i, k in enumerate(sa.SPURS)}


def _show(path: str, visible: bool) -> None:
    import isaaclab.sim as sim_utils
    from pxr import UsdGeom

    prim = sim_utils.get_current_stage().GetPrimAtPath(f"/World/envs/env_0/{path}")
    if prim.IsValid():
        img = UsdGeom.Imageable(prim)
        img.MakeVisible() if visible else img.MakeInvisible()


def _show_state(k: str, broken: str | None) -> None:
    """The tree and the apple as they look after `broken` (None: still whole)."""
    _show(f"tree/visual/spur_{k}", broken != "spur")
    _show(f"tree/visual/stub_{k}", broken == "spur")
    _show(f"tree/visual/stem_{k}", broken == "stem")
    _show(f"{NAMES[k]}/visual/stem", broken != "stem")
    _show(f"{NAMES[k]}/visual/torn", broken == "stem")
    _show(f"{NAMES[k]}/visual/spur", broken == "spur")


def _new_joints(env) -> None:
    _J.clear()
    tree_rot = np.eye(3)
    for k in sa.SPURS:
        centre, rot = sa.apple_rest(k)
        _, d = sa.spur_frame(k)
        j = sj.Joint(name=k, anchor=sa.ABSCISSION.copy(), point=sa.abscission_in_tree(k), rot0=rot, spur_dir=d,
                     spur_len=sa.SPUR_L, mass=sa.APPLE_MASS, inertia=0.4 * sa.APPLE_MASS * R_APPLE ** 2)
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


@on_reset
def _reset(env) -> None:
    """Failures stay failed until a real reset (the settling steps after it do not count)."""
    _clear()
    for k in sa.SPURS:
        if _J[k].broken is not None:  # cannot happen while settling; if it did, the trial starts with it
            _S["apple"][k]["broken"] = _J[k].broken


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
    tree_v, tree_w = _vel(env, "tree")
    tree_com = tree_pos + tree_rot @ np.array([0.0, 0.0, sa.TREE_COM_Z])
    total_f, total_t = np.zeros(3), np.zeros(3)
    for k in sa.SPURS:
        j: sj.Joint = _J[k]
        if j.broken is not None:
            if k not in _J["_zeroed"]:
                _set_wrench(env, NAMES[k], np.zeros(3), np.zeros(3))
                _J["_zeroed"].add(k)
            continue
        pos, rot = _pose(env, NAMES[k])
        v, w = _vel(env, NAMES[k])
        force, moment, p_tree, loads = j.wrench(pos, rot, v, w, tree_pos, tree_rot, tree_com, tree_v, tree_w, PHYSICS_DT)
        part = j.check(loads)
        if part is not None:
            j.broken = part
            _S["events"].append((round(_S["step"] / 20.0, 3), k, part, {n: round(x, 3) for n, x in loads.items()}))
            _set_wrench(env, NAMES[k], np.zeros(3), np.zeros(3))
            _J["_zeroed"].add(k)
            continue
        _set_wrench(env, NAMES[k], force, moment + np.cross(rot @ j.anchor, force))
        total_f -= force
        total_t += np.cross(p_tree - tree_com, -force) - moment
    _set_wrench(env, "tree", total_f, total_t)


def _tilt_deg(rot: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(rot[2, 2], -1.0, 1.0))))


@on_step
def _step(env) -> None:
    _S["step"] += 1
    tree_pos, tree_rot = _pose(env, "tree")
    if _tilt_deg(tree_rot) > POT_TIPPED_DEG:
        _S["tipped"] = True
    bucket_pos, _ = _pose(env, "bucket")
    touching = _touching()
    poses = {k: _pose(env, NAMES[k]) for k in sa.SPURS}
    for k in sa.SPURS:
        st = _S["apple"][k]
        j: sj.Joint = _J[k]
        centre, rot = poses[k]
        st["centre"], st["rot"] = centre, rot
        if j.broken is not None and st["broken"] is None:
            st["broken"] = j.broken
            _show_state(k, j.broken)
        others = [poses[o][0] for o in sa.SPURS if o != k and _S["apple"][o]["broken"] is not None]
        track_apple(st, centre, touching[k], others, bucket_pos, tree_pos)
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
           "events": _S["events"]}
    for k in sa.SPURS:
        st = _S["apple"][k]
        c, r = poses[k]
        row[k] = {"c": np.round(c, 4).tolist(), "x": np.round(r[:, 0], 3).tolist(), "broken": st["broken"],
                  "touch": bool(st["touched"]), "dropped": bool(st["dropped"]), "in": bool(st["in_bucket"]),
                  "drop_from": None if st["drop_from"] is None else round(float(st["drop_from"]), 4),
                  "peak": {n: round(x, 3) for n, x in _J[k].peak.items()}}
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
        h = np.array(hist)
        spread = float(np.linalg.norm(h[:, None] - h[None, :], axis=3).max())
    else:
        spread = math.inf
    touched = any(_S["apple"][k]["touched"] for k in sa.SPURS)
    out = dict(values=values, total=total, spread=spread, still=spread <= STILL_MOVE, touched=touched)
    _S["cache"] = (_S["step"], out)
    return out


def score(objects) -> float:
    """0 if the pot ever tips over. Otherwise each apple earns 0.5 (in the bucket, stem on, spur whole, not dropped) or
    0.25 (the same with its stem pulled out), and the sum is capped at 0.75: 1.0 comes only from success(), held 1 s."""
    return min(_evaluate()["total"], 0.75)


def success(objects) -> bool:
    """Both apples earn 0.5, no gripper touches either, and neither moves more than 5 mm over the last 1 s. The harness
    then needs it for 1 s in a row."""
    steptime.tick()  # the harness calls success() once per control step
    e = _evaluate()
    if os.environ.get("TASK_DEBUG") and _S["step"] % 20 == 0:
        a = _S["apple"]
        print(f"[task] t={_S['step'] / 20:.1f} total={e['total']} " + " ".join(
            f"{k}:{a[k]['broken']}/drop={a[k]['dropped']}/in={a[k]['in_bucket']}/touch={a[k]['touched']}" for k in sa.SPURS)
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


TWIST = math.radians(105.0)     # the hand turns the apple this far about its stem (the way that keeps the wrist in range) ...
TWIST_SIGN = {"a": 1.0, "b": -1.0}
LIFT = math.radians(12.0)       # ... while tilting it up this much and raising it 1 cm: the stem snaps at the spur
PULL = 0.10                     # the failure: the hand pulls this far straight away from the trunk
RETREAT = 0.06                  # after the stem snaps, the hand backs off this far before turning to the bucket
PLACE_GAP = 0.010               # the apple is let go with its lowest point this far above the pad
SLOTS = {"b": (-0.03, -0.042), "a": (-0.03, 0.042)}   # where each apple goes in the bucket, from its middle (x, y)
DOWN_YAW = math.pi / 2          # over the bucket the fingers close along x, so the second apple goes in beside the first
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


def _over_slot(k: str, z: float | None = None):
    def target(plan):
        p = _bucket_slot(k)
        return np.array([p[0], p[1], CARRY_Z if z is None else z])
    return target


def _slot(k: str):
    return lambda plan: _bucket_slot(k)


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


def pick(k: str, pull: bool, last: bool = False) -> list:
    """Pick apple k (lift and twist; or, for the failure, a straight pull) and lay it on the bucket's pad."""
    a = approach_dir(k)
    o, cl = FINGER_OPEN, FINGER_CLOSED
    level = level_grip(a)
    off = level_grip(a, 0.0 if pull else TWIST_SIGN[k] * TWIST, 0.0 if pull else LIFT)
    down = grip_rotation(DOWN_YAW)
    tilted = _ry(-math.radians(45.0)) @ down    # pointing forward and down: the arm cannot hold it straight down this high
    steep = _ry(-math.radians(12.0)) @ down
    zero = (0.0, 0.0, 0.0)
    w = [
        Joints(4.0, zero, o, f"in front of apple {k}", q=_joints_for(_from_apple(k, 0.10), level)),
        Pose(2.0, _from_apple(k, 0.0), o, f"round apple {k}", rot=level),
        Pose(0.8, _from_apple(k, 0.0), cl, f"close on apple {k}", rot=level),
        Pose(0.4, _from_apple(k, 0.0), cl, f"hold apple {k}", rot=level),
    ]
    if pull:
        w += [Pose(3.0, _from_apple(k, PULL), cl, f"pull apple {k} straight off", rot=level)]
        away = level
    else:
        w += [Pose(4.0, _from_apple(k, 0.0, 0.01), cl, f"lift and twist apple {k}", rot=off),
              Pose(0.5, _from_apple(k, 0.0, 0.01), cl, f"hold apple {k} off", rot=off),
              Pose(1.5, _from_apple(k, RETREAT, 0.02), cl, f"apple {k} away from the tree", rot=off)]
        away = off
    w += [
        Joints(4.0, zero, cl, f"apple {k} over the bucket", q=_joints_for(_over_slot(k), tilted)),
        Pose(1.5, _over_slot(k, MID_Z), cl, f"apple {k} into the bucket", rot=steep),
        Pose(1.5, _slot(k), cl, f"lower apple {k} to the pad", rot=down),
        Pose(0.8, _slot(k), o, f"let go of apple {k}", rot=down),
    ]
    if last:  # out of the camera's way quickly: the trial ends 2 s after both apples are still
        w += [Pose(0.5, _over_slot(k, MID_Z), o, f"up from apple {k}", rot=steep),
              Pose(0.6, _over_slot(k), o, f"out of the bucket ({k})", rot=tilted),
              Joints(1.2, zero, o, "back to the start", q=tuple(READY))]
    else:
        w += [Pose(1.2, _over_slot(k, MID_Z), o, f"up from apple {k}", rot=steep),
              Pose(1.2, _over_slot(k), o, f"out of the bucket ({k})", rot=tilted)]
    del away
    return w


def demo():
    """The right arm picks both apples, the upper one first, and lays each on the bucket's pad. TASK_DEMO=failure pulls
    each apple straight away from the trunk instead (the stem tears out of the apple), then lays it in the same way."""
    pull = os.environ.get("TASK_DEMO") == "failure"
    w = pick("b", pull) + pick("a", pull, last=True)
    w += [Joints(4.0, (0.0, 0.0, 0.0), FINGER_OPEN, "wait", q=tuple(READY))]
    return {"right": w}
