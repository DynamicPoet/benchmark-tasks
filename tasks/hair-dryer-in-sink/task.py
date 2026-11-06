"""On the station table: a white ceramic sink filled with water, a chrome tap behind it. A red hair dryer lies on its side in
the water, its handle towards the arms. Its cord crosses the sink's front rim, runs along the table and rises to a black plug in
a wall socket on a low tiled splashback, beside the sink on the left arm's side. A folded blue towel lies on the table on the
right arm's side.

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
from scenekit import UsdBody, on_reset, on_step  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP, TCP_OFFSET  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

sa.write_all()

_pos, _rot = sa.dryer_pose(TOP)
DRYER = UsdBody("dryer", usd="assets/dryer.usda", pos=_pos, rot_xyzw=_rot, size=(0.2, 0.2, 0.07), articulation=True,
                joint_pos=((".*", 0.0),), joint_damping=2.0e-4, joint_armature=1.0e-5, friction=(0.9, 0.8), solver_iters=32)
SINK = UsdBody("sink", usd="assets/sink.usda", pos=(*sa.SINK_XY, TOP), size=(*sa.SINK_SIZE, sa.SINK_H), friction=(0.5, 0.4))
WALL = UsdBody("wall", usd="assets/wall.usda", pos=(sa.SOCKET[0], sa.SOCKET[1], TOP), size=sa.WALL_SIZE, friction=(0.5, 0.4))
TOWEL = UsdBody("towel", usd="assets/towel.usda", pos=(*sa.TOWEL_XY, TOP), size=sa.TOWEL_SIZE, friction=(0.9, 0.8))
OBJECTS = [SINK, WALL, TOWEL, DRYER]

# -- what the scorer measures ------------------------------------------------------------------------------------------

PLUG_OUT = 0.02       # the pins are out when the plug has moved this far from its seat
PLUG_HOLD_N = 5.0     # how hard the socket's contacts hold the plug, until the pins are out
NEAR_PLUG = 0.03      # a gripper's fingertip this close to the plug's body counts as having pulled it
DRYER_MOVED = 0.02    # how far the dryer may move while the plug is in
CLEAR = 0.05          # every fingertip this far from the dryer at the end
REST_MOVE = 0.005     # at rest: moved less than this ...
REST_TURN = math.radians(3.0)  # ... and turned less than this ...
REST_STEPS = 20       # ... over the last second at 20 Hz
ON_TOWEL_GAP = 0.015  # the dryer's lowest point within this of the towel's top: it rests there
TIP_IN_GRIPPER = np.array([0.0, 0.0, TCP_OFFSET[2] + PAD_TIP_PAST_TCP])

_seat = np.array([sa.PLUG_MID[0], sa.PLUG_MID[1], TOP + sa.PLUG_MID[2]])
_sink_in = (np.array(sa.SINK_SIZE) - 2 * sa.SINK_WALL) / 2
_sink_out = np.array(sa.SINK_SIZE) / 2
_sink_xy = np.array(sa.SINK_XY)
_towel_half = np.array(sa.TOWEL_SIZE[:2]) / 2
_towel_xy = np.array(sa.TOWEL_XY)
_towel_top = TOP + sa.TOWEL_SIZE[2]
_barrel = (np.zeros(3), np.array([sa.BARREL_R, sa.BARREL_L / 2, sa.BARREL_R]))
_handle = (np.array([(sa.HANDLE[0] + sa.HANDLE[1]) / 2, 0.0, 0.0]),
           np.array([(sa.HANDLE[1] - sa.HANDLE[0]) / 2, sa.HANDLE_Y / 2, sa.HANDLE_Z / 2]))
_plug_body = (np.zeros(3), np.array(sa.PLUG_SIZE) / 2)

_S: dict = {}


def _clear() -> None:
    _S.clear()
    _S.update(in_basin=False, moved=False, yanked=False, fell=False, plug_in=True, plug_gap=0.0, start=None, history=[],
              tip_to_dryer=9.9, dryer_low=TOP, dryer_pos=None)


def _np(x) -> np.ndarray:
    return getattr(x, "torch", x).detach().cpu().numpy().astype(np.float64)


def _rotation(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _to_box(points: np.ndarray, pos: np.ndarray, rot: np.ndarray, box) -> float:
    """The smallest distance from the points to a box (centre and half sizes in the body's frame)."""
    local = (points - pos) @ rot - box[0]
    return float(np.linalg.norm(np.maximum(np.abs(local) - box[1], 0.0), axis=1).min())


def _fingertips(env) -> np.ndarray:
    """The middle of each pad's tip, four points, in the table's frame."""
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


def _hold_plug(env, pushing: bool) -> None:
    """The socket's contacts, as a steady push towards the wall until the pins are out."""
    import torch

    asset = env.unwrapped.scene["dryer"]
    i = asset.body_names.index("plug")
    force = torch.zeros((1, 1, 3), device=asset.device)
    if pushing:
        force[0, 0, 0] = PLUG_HOLD_N
    asset.permanent_wrench_composer.set_forces_and_torques_index(forces=force, torques=torch.zeros_like(force),
                                                                 body_ids=torch.tensor([i], device=asset.device), is_global=True)


def _measure(env) -> None:
    base = env.unwrapped
    asset = base.scene["dryer"]
    origin = _np(base.scene.env_origins[0])
    pos = _np(asset.data.root_link_pose_w)[0, :3] - origin
    quat = _np(asset.data.root_link_pose_w)[0, 3:7]
    rot = _rotation(quat)
    i = asset.body_names.index("plug")
    plug_pos = _np(asset.data.body_pos_w)[0, i] - origin
    plug_rot = _rotation(_np(asset.data.body_quat_w)[0, i])
    tips = _fingertips(env)

    if _S.get("start") is None:
        _clear()
        _S["start"] = pos.copy()
    gap = float(np.linalg.norm(plug_pos - _seat))
    was_in = _S["plug_in"]
    now_in = gap < PLUG_OUT
    _hold_plug(env, now_in)
    if was_in and not now_in and _to_box(tips, plug_pos, plug_rot, _plug_body) > NEAR_PLUG:
        _S["yanked"] = True
    if now_in:
        inside = np.all(np.abs(tips[:, :2] - _sink_xy) < _sink_in, axis=1) & (tips[:, 2] < TOP + sa.SINK_H)
        _S["in_basin"] = _S["in_basin"] or bool(inside.any())
        _S["moved"] = _S["moved"] or float(np.linalg.norm(pos - _S["start"])) > DRYER_MOVED
    corners = np.array([[hx, sy * sa.HANDLE_Y / 2, sz * sa.HANDLE_Z / 2] for hx in sa.HANDLE for sy in (-1, 1) for sz in (-1, 1)])
    ends = np.array([[0.0, sy * sa.BARREL_L / 2, 0.0] for sy in (-1, 1)])
    low = min(float((pos + corners @ rot.T)[:, 2].min()), float((pos + ends @ rot.T)[:, 2].min()) - sa.BARREL_R)
    _S["fell"] = _S["fell"] or pos[2] < TOP - 0.02
    _S["history"] = (_S["history"] + [(pos.copy(), rot.copy())])[-(REST_STEPS + 1):]
    _S.update(plug_in=now_in, plug_gap=gap, dryer_low=low, dryer_pos=pos,
              tip_to_dryer=min(_to_box(tips, pos, rot, _barrel), _to_box(tips, pos, rot, _handle)))


@on_reset
def _on_reset(env) -> None:
    _clear()
    _measure(env)


@on_step
def _on_step(env) -> None:
    _measure(env)


def _at_rest() -> bool:
    h = _S["history"]
    if len(h) <= REST_STEPS:
        return False
    moved = float(np.linalg.norm(h[-1][0] - h[0][0]))
    cos_turn = (np.trace(h[0][1].T @ h[-1][1]) - 1.0) / 2.0
    return moved < REST_MOVE and math.acos(float(np.clip(cos_turn, -1.0, 1.0))) < REST_TURN


def score(objects) -> float:
    """0 for any failure, at any moment. Otherwise 0.25 plug out; 0.5 dryer out of the sink; 0.75 on the towel; 1.0 let go and
    still. `task.json` states every condition."""
    if not _S or _S["in_basin"] or _S["moved"] or _S["yanked"] or _S["fell"] or _S["plug_in"]:
        return 0.0
    pos, low = _S["dryer_pos"], _S["dryer_low"]
    over_sink = bool(np.all(np.abs(pos[:2] - _sink_xy) < _sink_out))
    if over_sink and low < TOP + sa.SINK_H:
        return 0.25
    on_towel = bool(np.all(np.abs(pos[:2] - _towel_xy) < _towel_half)) and low < _towel_top + ON_TOWEL_GAP
    if not on_towel:
        return 0.5
    return 1.0 if _S["tip_to_dryer"] >= CLEAR and _at_rest() else 0.75


def success(objects) -> bool:
    steptime.tick()  # the harness calls success() once per control step
    return score(objects) == 1.0


def distance(objects) -> float:
    return float(np.linalg.norm(objects["dryer"].pos[:2] - _towel_xy))


# -- the scripted demo -------------------------------------------------------------------------------------------------

PLUG_GRIP_X = 0.252   # the pads on the plug's body; the fingers widen higher up and must clear the plate
HANDLE_GRIP = (0.315, -0.03)
TOWEL_GRIP = (0.205, -0.33)  # puts the barrel's middle over the towel's middle
LEFT_CLEAR = (0.14, 0.305, TOP + 0.16)
RIGHT_CLEAR = (0.14, -0.305, TOP + 0.16)


def _unplug() -> list[Waypoint]:
    x, y, z = PLUG_GRIP_X, sa.SOCKET[1], TOP + sa.SOCKET[2]
    out = x - 0.055
    return [
        Waypoint(1.5, (0.20, y, TOP + 0.17), FINGER_OPEN, "back from the wall"),
        Waypoint(1.0, (x, y, TOP + 0.17), FINGER_OPEN, "above the plug"),
        Waypoint(1.2, (x, y, z), FINGER_OPEN, "down to the plug"),
        Waypoint(0.8, (x, y, z), FINGER_CLOSED, "pinch"),
        Waypoint(1.5, (out, y, z), FINGER_CLOSED, "pull it out"),
        Waypoint(0.4, (out, y, z), FINGER_CLOSED, "hold"),
        Waypoint(0.6, (out, y, z), FINGER_OPEN, "let go"),
        Waypoint(1.0, (out, y, TOP + 0.17), FINGER_OPEN, "rise"),
        Waypoint(1.5, LEFT_CLEAR, FINGER_OPEN, "clear"),
    ]


def _fetch() -> list[Waypoint]:
    x, y = HANDLE_GRIP
    tx, ty = TOWEL_GRIP
    grip, travel = TOP + 0.04, TOP + 0.145
    release = TOP + sa.TOWEL_SIZE[2] + 0.04
    return [
        Waypoint(1.5, (x, y, travel), FINGER_OPEN, "above the handle"),
        Waypoint(1.5, (x, y, grip), FINGER_OPEN, "into the water"),
        Waypoint(0.8, (x, y, grip), FINGER_CLOSED, "grip"),
        Waypoint(1.5, (x, y, travel), FINGER_CLOSED, "lift out"),
        Waypoint(2.5, (tx, ty, travel), FINGER_CLOSED, "carry"),
        Waypoint(1.2, (tx, ty, release), FINGER_CLOSED, "lower"),
        Waypoint(0.8, (tx, ty, release), FINGER_OPEN, "let go"),
        Waypoint(1.0, (tx, ty, travel), FINGER_OPEN, "rise"),
        Waypoint(1.5, RIGHT_CLEAR, FINGER_OPEN, "clear"),
        Waypoint(4.0, RIGHT_CLEAR, FINGER_OPEN, "watch"),
    ]


def demo() -> dict[str, list[Waypoint]]:
    """TASK_DEMO=failure reaches into the water with the plug still in. Open loop, from the starting layout only."""
    if os.environ.get("TASK_DEMO") == "failure":
        return {"right": _fetch()}
    left = _unplug()
    wait = sum(w.duration for w in left)
    return {"left": left, "right": [Waypoint(wait, RIGHT_CLEAR, FINGER_OPEN, "wait"), *_fetch()]}
