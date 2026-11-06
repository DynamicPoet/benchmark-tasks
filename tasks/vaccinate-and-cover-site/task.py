"""On the station table: an injection-training arm lying on a cushion in front of the robot, its upper arm along y, a dark circle
printed on it. To its left, a pre-filled injector pen standing in a holder, and behind that a yellow sharps bin. To its right, a
square gauze pad on a small tray. The pen injects when it is pressed against the skin; the printed circle has a spring-loaded pad
under it that gives way when pressed. When the pen comes out, a red spot appears at the site and grows until it is covered.

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

ARM = UsdBody("arm", usd="assets/arm.usda", pos=(*sa.TARGET_XY, TOP), size=(0.1, 0.3, 0.1), friction=(0.8, 0.7))
PAD = UsdBody("skin_pad", usd="assets/skin_pad.usda", pos=(*sa.TARGET_XY, TOP), size=(0.04, 0.04, 0.02), articulation=True, joint_pos=(("skin", 0.0),),
              joint_groups=(("skin", 25.0, sa.PAD_STIFFNESS),), friction=(0.8, 0.7), solver_iters=32)
PEN_START = (sa.HOLDER_XY[0], sa.HOLDER_XY[1], TOP + sa.HOLDER_FLOOR + sa.PEN_L / 2 + 0.001)
PEN = UsdBody("pen", usd="assets/pen.usda", pos=PEN_START, size=(sa.PEN_W, sa.PEN_W, sa.PEN_L), mass=sa.PEN_MASS, friction=(0.9, 0.8), solver_iters=32, torsional_patch=0.01)
HOLDER = UsdBody("holder", usd="assets/holder.usda", pos=(*sa.HOLDER_XY, TOP), size=sa.HOLDER, friction=(0.5, 0.4))
GAUZE_START = (sa.TRAY_XY[0], sa.TRAY_XY[1], TOP + sa.TRAY[2] + sa.GAUZE[2] / 2 + 0.0005)
GAUZE = UsdBody("gauze", usd="assets/gauze.usda", pos=GAUZE_START, size=sa.GAUZE, mass=sa.GAUZE_MASS, friction=(0.9, 0.8), solver_iters=32, torsional_patch=0.012)
TRAY = UsdBody("tray", usd="assets/tray.usda", pos=(*sa.TRAY_XY, TOP), size=sa.TRAY, friction=(0.5, 0.4))
BIN = UsdBody("sharps_bin", usd="assets/bin.usda", pos=(*sa.BIN_XY, TOP), size=sa.BIN, friction=(0.5, 0.4))
OBJECTS = [ARM, PAD, HOLDER, TRAY, BIN, PEN, GAUZE]

CONTROL_DT = 0.05
PRESSED_DEPTH = 0.01    # the pen has the skin pad in by this much
DOSE_S = 5.0            # the pen must stay pressed this long
COVER_WITHIN_S = 4.0    # the gauze must land this soon after the pen lifts
COVER_PRESS_S = 3.0     # and be pressed this long
GAUZE_PRESS_DEPTH = 0.001
ON_SKIN = 0.004         # a lowest point this close above the skin is on it
TOUCH = 0.002           # the pen's tip this close to the arm touches it
LIFTED = 0.005          # the tip is off the skin once this far above it
CLEAR = 0.05            # every fingertip this far from the arm at the end
ARM_MOVED = 0.01
FOREARM_TOP = 0.08
REST_MOVE = 0.005       # the gauze moves less than this ...
REST_STEPS = 20         # ... over the last second
TIP_IN_GRIPPER = np.array([0.0, 0.0, TCP_OFFSET[2] + PAD_TIP_PAST_TCP])
SPOT_RADII_S = (0.0, 1.5, 3.0, 5.0, 8.0)  # seconds after the pen lifts, when each size of red spot appears

_target = np.array([sa.TARGET_XY[0], sa.TARGET_XY[1], TOP + sa.PAD_TOP])
_S: dict = {}


def _clear() -> None:
    _S.clear()
    _S.update(ready=False, failed=None, press_s=0.0, pressed_ever=False, dose=False, lifted_at=None, t=0.0, covered_at=None, cover_press_s=0.0, cover_press_best=0.0,
              covered=False, pen_in_bin=False, clear=False, history=[], spots=-1, covered_ever=False)


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


def _body(env, name: str) -> tuple[np.ndarray, np.ndarray]:
    base = env.unwrapped
    origin = _np(base.scene.env_origins[0])
    pose = _np(base.scene[name].data.root_link_pose_w)[0]
    return pose[:3] - origin, _rotation(pose[3:7])


def _to_box(points: np.ndarray, centre: np.ndarray, half: np.ndarray) -> float:
    return float(np.linalg.norm(np.maximum(np.abs(points - centre) - half, 0.0), axis=1).min())


def _show_spots(n: int) -> None:
    """The red spot, as a stack of discs: the first n show. Only for the eye; the score does not use it."""
    if n == _S["spots"]:
        return
    _S["spots"] = n
    import isaaclab.sim as sim_utils
    from pxr import UsdGeom

    stage = sim_utils.get_current_stage()
    for k in range(5):
        prim = stage.GetPrimAtPath(f"/World/envs/env_0/arm/visual/spot_{k}")
        if prim.IsValid():
            image = UsdGeom.Imageable(prim)
            image.MakeVisible() if k < n else image.MakeInvisible()


def _fail(reason: str) -> None:
    if _S["failed"] is None:
        _S["failed"] = reason


def _measure(env) -> None:
    if not _S.get("ready"):
        _clear()
        _S["ready"] = True
    pen_pos, pen_rot = _body(env, "pen")
    gauze_pos, gauze_rot = _body(env, "gauze")
    tip = pen_pos + pen_rot @ np.array([0.0, 0.0, -sa.PEN_L / 2])
    tips = _fingertips(env)
    pad = env.unwrapped.scene["skin_pad"]
    depth = -float(_np(pad.data.joint_pos)[0, 0])
    _S["t"] += CONTROL_DT

    off_centre = float(np.linalg.norm(tip[:2] - _target[:2]))
    on_arm_y = abs(tip[1] - _target[1]) < 0.19 and abs(tip[0] - _target[0]) < 0.036
    touching = on_arm_y and tip[2] - (TOP + sa.SKIN_TOP) < TOUCH + (sa.PAD_PROUD if off_centre < sa.PAD_HALF else 0.0)
    on_forearm = _target[0] - 0.37 < tip[0] < _target[0] + 0.04 and _target[1] - 0.20 < tip[1] < _target[1] - 0.11
    touching = touching or (on_forearm and tip[2] - (TOP + FOREARM_TOP) < TOUCH)
    pressing = off_centre <= sa.TARGET_R and depth >= PRESSED_DEPTH
    if touching and off_centre > sa.TARGET_R:
        _fail("pen touched the arm outside the circle")
    if touching and _S["lifted_at"] is not None:
        _fail("pen touched the arm a second time")
    if pressing:
        _S["pressed_ever"] = True
        _S["press_s"] += CONTROL_DT
        _S["dose"] = _S["dose"] or _S["press_s"] >= DOSE_S
    elif _S["pressed_ever"] and _S["lifted_at"] is None and not pressing and tip[2] - _target[2] > LIFTED:
        _S["lifted_at"] = _S["t"]
    elif _S["pressed_ever"] and _S["lifted_at"] is None and not pressing and not _S["dose"]:
        _S["press_s"] = 0.0  # a press must be without a break

    low = gauze_pos[2] - sa.GAUZE[2] / 2
    near_site = float(np.linalg.norm(gauze_pos[:2] - _target[:2]))
    on_skin = low - _target[2] < ON_SKIN
    if on_skin and near_site < sa.GAUZE[0] / 2 + sa.TARGET_R and not (_S["dose"] and _S["lifted_at"] is not None):
        _fail("gauze touched the site before the dose was given")
    covered = near_site <= sa.TARGET_R and on_skin and _S["lifted_at"] is not None
    if covered and _S["covered_at"] is None:
        _S["covered_at"] = _S["t"]
    if covered and depth >= GAUZE_PRESS_DEPTH:
        _S["cover_press_s"] += CONTROL_DT
        _S["cover_press_best"] = max(_S["cover_press_best"], _S["cover_press_s"])
    else:
        _S["cover_press_s"] = 0.0
    if pen_pos[2] < TOP - 0.02 or gauze_pos[2] < TOP - 0.02:
        _fail("the pen or the gauze fell off the table")
    if float(np.linalg.norm(_body(env, "arm")[0] - np.array(ARM.pos))) > ARM_MOVED:
        _fail("the training arm moved")

    half = np.array(sa.BIN[:2]) / 2
    pen_in_bin = bool(np.all(np.abs(pen_pos[:2] - np.array(sa.BIN_XY)) < half)) and pen_pos[2] < TOP + sa.BIN[2]
    arm_tip_gap = min(_to_box(tips, np.array([_target[0], _target[1] + 0.025, TOP + 0.05]), np.array([0.05, 0.165, 0.05])),
                      _to_box(tips, np.array([_target[0] - 0.165, _target[1] - 0.155, TOP + 0.04]), np.array([0.205, 0.05, 0.04])))
    _S["history"] = (_S["history"] + [gauze_pos.copy()])[-(REST_STEPS + 1):]
    _S.update(covered=covered, pen_in_bin=pen_in_bin, clear=bool(arm_tip_gap >= CLEAR), depth=depth, tip=tip)
    # The red spot: from the pen's lifting until it is covered
    if _S["covered_at"] is not None or _S["lifted_at"] is None:
        _show_spots(0)
    else:
        elapsed = _S["t"] - _S["lifted_at"]
        _show_spots(sum(1 for s in SPOT_RADII_S if elapsed >= s))


@on_reset
def _on_reset(env) -> None:
    _clear()
    _show_spots(0)
    _S["ready"] = True


@on_step
def _on_step(env) -> None:
    _measure(env)


def _at_rest() -> bool:
    h = _S["history"]
    return len(h) > REST_STEPS and float(np.linalg.norm(h[-1] - h[0])) < REST_MOVE


def _score() -> float:
    """0 for any failure, at any moment, or if the pen never pressed the target. `task.json` states every condition."""
    if not _S or _S["failed"] or not _S["pressed_ever"]:
        return 0.0
    if not _S["dose"]:
        return 0.25
    if _S["lifted_at"] is None or not _S["covered"]:
        return 0.5
    late = _S["covered_at"] - _S["lifted_at"] > COVER_WITHIN_S
    short = _S["cover_press_best"] < COVER_PRESS_S
    if late or short or not _S["pen_in_bin"]:
        return 0.75
    return 1.0 if _S["clear"] and _at_rest() else 0.75


def score(objects) -> float:
    return _score()


def success(objects) -> bool:
    steptime.tick()  # the harness calls success() once per control step
    return _score() == 1.0


def distance(objects) -> float:
    return float(np.linalg.norm(objects["gauze"].pos[:2] - _target[:2]))


# -- the scripted demo -------------------------------------------------------------------------------------------------

LEFT_CLEAR = (0.14, 0.305, TOP + 0.16)
RIGHT_CLEAR = (0.14, -0.305, TOP + 0.16)
ACROSS = {1: -math.pi / 2, -1: math.pi / 2}  # the grip that closes the fingers along x, per arm
GRIP_ABOVE_TIP = 0.082  # TCP above the pen's tip: the pen's top clears the gripper's palm
PEN_OPEN = -0.024
GAUZE_OPEN = -0.036
READY_OFF = 0.17  # the gauze waits this far from the site: nearer, the two wrists meet


def _left_timeline() -> tuple[list[Waypoint], float]:
    yaw = ACROSS[1]
    hx, hy = sa.HOLDER_XY
    tx, ty = sa.TARGET_XY
    grip = sa.HOLDER_FLOOR + GRIP_ABOVE_TIP
    hover = 0.19
    press = sa.PAD_TOP + GRIP_ABOVE_TIP - 0.026  # the soft arm lags by about a centimetre under the spring
    plan = [
        Waypoint(1.5, (hx - 0.02, hy + 0.06, TOP + 0.175), PEN_OPEN, "beside the pen", yaw),
        Waypoint(1.0, (hx, hy, TOP + 0.175), PEN_OPEN, "above the pen", yaw),
        Waypoint(1.2, (hx, hy, TOP + grip), PEN_OPEN, "down to the pen", yaw),
        Waypoint(0.8, (hx, hy, TOP + grip), FINGER_CLOSED, "grip", yaw),
        Waypoint(1.2, (hx, hy, TOP + hover), FINGER_CLOSED, "lift out", yaw),
        Waypoint(1.5, (tx, ty, TOP + hover), FINGER_CLOSED, "over the target", yaw),
        Waypoint(1.2, (tx, ty, TOP + press), FINGER_CLOSED, "press", yaw),
        Waypoint(5.8, (tx, ty, TOP + press), FINGER_CLOSED, "hold the dose", yaw),
    ]
    lifted_at = sum(w.duration for w in plan) + 0.4
    plan += [
        Waypoint(1.0, (tx, ty, TOP + hover), FINGER_CLOSED, "lift", yaw),
        Waypoint(1.5, (sa.BIN_XY[0], sa.BIN_XY[1], TOP + 0.205), FINGER_CLOSED, "over the bin", yaw),
        Waypoint(0.6, (sa.BIN_XY[0], sa.BIN_XY[1], TOP + 0.205), PEN_OPEN, "drop it", yaw),
        Waypoint(1.0, (sa.BIN_XY[0], sa.BIN_XY[1], TOP + 0.215), PEN_OPEN, "wait", yaw),
        Waypoint(1.2, LEFT_CLEAR, FINGER_OPEN, "clear"),
    ]
    return plan, lifted_at


def _right_timeline(lifted_at: float, cover: bool) -> list[Waypoint]:
    yaw = ACROSS[-1]
    gx, gy = sa.TRAY_XY
    tx, ty = sa.TARGET_XY
    grip = sa.TRAY[2] + sa.GAUZE[2] / 2 + 0.0005  # TCP level with the gauze's middle: the pads' flat part, the tips below it
    plan = [
        Waypoint(1.5, (gx, gy, TOP + 0.13), GAUZE_OPEN, "above the gauze", yaw),
        Waypoint(1.2, (gx, gy, TOP + grip), GAUZE_OPEN, "down to the gauze", yaw),
        Waypoint(0.8, (gx, gy, TOP + grip), FINGER_CLOSED, "grip", yaw),
        Waypoint(1.0, (gx, gy, TOP + 0.13), FINGER_CLOSED, "lift", yaw),
        Waypoint(1.2, (tx, ty - READY_OFF, TOP + 0.14), FINGER_CLOSED, "ready beside the site", yaw),
    ]
    wait = lifted_at + 0.2 - sum(w.duration for w in plan)
    plan.append(Waypoint(max(wait, 0.1), (tx, ty - READY_OFF, TOP + 0.14), FINGER_CLOSED, "wait for the pen", yaw))
    if not cover:
        plan.append(Waypoint(12.0, (tx, ty - READY_OFF, TOP + 0.14), FINGER_CLOSED, "does nothing", yaw))
        return plan
    on_skin = sa.PAD_TOP + sa.GAUZE[2] / 2
    press = on_skin - 0.004
    plan += [
        Waypoint(1.4, (tx, ty, TOP + 0.14), FINGER_CLOSED, "over the site", yaw),
        Waypoint(0.8, (tx, ty, TOP + press), FINGER_CLOSED, "cover it", yaw),
        Waypoint(3.6, (tx, ty, TOP + press), FINGER_CLOSED, "hold it on", yaw),
        Waypoint(0.6, (tx, ty, TOP + on_skin), GAUZE_OPEN, "let go", yaw),
        Waypoint(1.0, (tx, ty, TOP + 0.15), GAUZE_OPEN, "rise", yaw),
        Waypoint(1.2, RIGHT_CLEAR, FINGER_OPEN, "clear"),
        Waypoint(4.0, RIGHT_CLEAR, FINGER_OPEN, "watch"),
    ]
    return plan


def demo() -> dict[str, list[Waypoint]]:
    """TASK_DEMO=failure holds the gauze ready and does nothing. Open loop, with timed waits only."""
    left, lifted_at = _left_timeline()
    return {"left": left, "right": _right_timeline(lifted_at, os.environ.get("TASK_DEMO") != "failure")}
