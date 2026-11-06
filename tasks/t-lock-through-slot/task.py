"""On the station table, in front of the robot: a blue-grey painted steel panel, 40 cm wide and 12 cm tall, fixed to the table across
the arms' view, with a narrow vertical slot in its middle. A steel rod runs through the slot along x. On the robot's side it ends in
an orange T-handle. Behind the panel it ends in a T-shaped brass head lying across the slot, which turns freely on the rod, with a
steel crank pin, a free-spinning sleeve, standing on its back near its right end.

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

PANEL = UsdBody("panel", usd="assets/panel.usda", pos=(sa.PANEL_X, 0.0, TOP), size=(sa.PANEL_T, sa.PANEL_W, sa.PANEL_H), friction=(0.5, 0.4))
ROD_SPRING, HEAD_SPRING = 30.0, 0.05  # N/m, N m/rad: light springs, back to the start
LOCK = UsdBody("t_lock", usd="assets/lock.usda", pos=(sa.HEAD_X, 0.0, TOP + sa.ROD_Z), size=(0.2, 0.1, 0.1), articulation=True, joint_pos=((".*", 0.0),),
               joint_groups=(("rod_slide", 6.0, ROD_SPRING), ("head_turn", 0.02, HEAD_SPRING), ("pin_spin", 1.0e-4, 0.0)), joint_armature=1.0e-5, friction=(0.7, 0.6), solver_iters=32)
OBJECTS = [PANEL, LOCK]

UPRIGHT_DEG = 10.0    # within this of upright counts
CLEAR = 0.05          # every fingertip this far from the handle, the rod and the head at the end
PANEL_MOVED = 0.005
REST_MOVE = 0.005     # the handle moves less than this ...
REST_STEPS = 20       # ... over the last second
CATCH_STIFFNESS = 3000.0
TIP_IN_GRIPPER = np.array([0.0, 0.0, TCP_OFFSET[2] + PAD_TIP_PAST_TCP])

_S: dict = {}


def _clear() -> None:
    _S.clear()
    _S.update(ready=False, upright=False, in_slot=False, caught=False, panel_moved=False, travel=0.0, head_deg=0.0, clear=False, history=[], near=False)


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


def _catch(asset) -> None:
    """The catch at the end of the rod's travel: it holds the rod there."""
    import torch

    i = asset.joint_names.index("rod_slide")
    now = asset.data.joint_pos
    now = getattr(now, "torch", now)[0, [i]].clone().view(1, -1)
    asset.set_joint_position_target_index(target=now, joint_ids=[i])
    asset.write_joint_stiffness_to_sim_index(stiffness=torch.tensor([[CATCH_STIFFNESS]], device=now.device, dtype=torch.float32), joint_ids=[i])


def _release(asset) -> None:
    import torch

    i = asset.joint_names.index("rod_slide")
    device = asset.device
    asset.set_joint_position_target_index(target=torch.zeros((1, 1), device=device), joint_ids=[i])
    asset.write_joint_stiffness_to_sim_index(stiffness=torch.tensor([[ROD_SPRING]], device=device, dtype=torch.float32), joint_ids=[i])


def _to_box(points: np.ndarray, centre: np.ndarray, half: np.ndarray, rot: np.ndarray | None = None) -> float:
    local = points - centre
    if rot is not None:
        local = local @ rot
    return float(np.linalg.norm(np.maximum(np.abs(local) - half, 0.0), axis=1).min())


def _measure(env) -> None:
    base = env.unwrapped
    asset = base.scene["t_lock"]
    if not _S.get("ready"):
        _clear()
        _release(asset)
        _S["ready"] = True
    origin = _np(base.scene.env_origins[0])
    q = _np(asset.data.joint_pos)[0]
    travel = -float(q[asset.joint_names.index("rod_slide")])
    head_deg = math.degrees(float(q[asset.joint_names.index("head_turn")]))
    tips = _fingertips(env)
    panel = _np(base.scene["panel"].data.root_link_pose_w)[0, :3] - origin
    _S["panel_moved"] = _S["panel_moved"] or float(np.linalg.norm(panel - np.array(PANEL.pos))) > PANEL_MOVED
    _S["upright"] = _S["upright"] or abs(-head_deg - sa.UPRIGHT_Q_DEG) <= UPRIGHT_DEG
    _S["in_slot"] = _S["in_slot"] or travel >= sa.HEAD_IN_SLOT
    if not _S["caught"] and travel >= sa.AT_STOP:
        _S["caught"] = True
        _catch(asset)
    i_rod = asset.body_names.index("rod")
    rod_pos = _np(asset.data.body_pos_w)[0, i_rod] - origin
    i_head = asset.body_names.index("head")
    head_pos = _np(asset.data.body_pos_w)[0, i_head] - origin
    head_rot = _rotation(_np(asset.data.body_quat_w)[0, i_head])
    rod_centre = rod_pos + np.array([(sa.BAR_X - sa.HEAD_X) / 2, 0.0, 0.0])
    rod_half = np.array([(sa.HEAD_X - sa.BAR_X) / 2 + sa.BAR_R, sa.BAR_L / 2, sa.BAR_R])
    gap_handle = _to_box(tips, rod_centre, rod_half)
    gap_head = _to_box(tips, head_pos + head_rot @ np.array([sa.PIN_L / 2, 0.0, 0.0]), np.array([sa.HEAD_T / 2 + sa.PIN_L / 2, sa.HEAD_L / 2, sa.HEAD_W / 2]), head_rot)
    _S["history"] = (_S["history"] + [travel])[-(REST_STEPS + 1):]
    gap = min(gap_handle, gap_head)
    _S.update(travel=travel, head_deg=head_deg, clear=gap >= CLEAR, near=gap_handle < CLEAR)


@on_reset
def _on_reset(env) -> None:
    _clear()
    _release(env.unwrapped.scene["t_lock"])
    _S["ready"] = True


@on_step
def _on_step(env) -> None:
    _measure(env)


def _at_rest() -> bool:
    h = _S["history"]
    return len(h) > REST_STEPS and abs(h[-1] - h[0]) < REST_MOVE


def score(objects) -> float:
    """0 if the panel moves. Otherwise the highest reached: 0.25 head upright; 0.5 head in the slot; at the end, 0.75 at the stop; 1.0
    at the stop, grippers clear and still."""
    if not _S or _S["panel_moved"]:
        return 0.0
    if _S["travel"] >= sa.AT_STOP:
        return 1.0 if _S["clear"] and _at_rest() else 0.75
    if _S["in_slot"]:
        return 0.5
    return 0.25 if _S["upright"] else 0.0


def success(objects) -> bool:
    steptime.tick()  # the harness calls success() once per control step
    return score(objects) == 1.0


def distance(objects) -> float:
    return max(0.0, sa.AT_STOP - _S.get("travel", 0.0))


# -- the scripted demo -------------------------------------------------------------------------------------------------

LEFT_CLEAR = (0.14, 0.305, TOP + 0.16)
RIGHT_CLEAR = (0.14, -0.305, TOP + 0.16)
OVER_PANEL = 0.148
STRADDLE = -0.016
GRIP_X = 0.19
PULL_INTO_SLOT = 0.048
PULL_TO_STOP = 0.104


def demo() -> dict[str, list[Waypoint]]:
    """TASK_DEMO=failure pulls alone. Open loop, with timed waits only."""
    z = TOP + sa.ROD_Z
    pin_x = sa.HEAD_X + sa.HEAD_T / 2 + sa.PIN_L * 0.6
    grip_z = z + 0.010  # the tips below the pin as it starts, 20 degrees up
    tilt = math.radians(sa.REST_TILT_DEG)
    start_y = -sa.PIN_OFFSET * math.cos(tilt) - sa.PIN_W / 2 - 0.003 + abs(STRADDLE)  # the near finger just beyond the pin
    upright_y = -sa.PIN_W / 2 + 0.016 + abs(STRADDLE)  # well past upright: the soft wrist gives way, and the head rests on its stop
    right = [
        Waypoint(1.2, (0.20, -0.20, TOP + OVER_PANEL + 0.01), STRADDLE, "towards the panel"),
        Waypoint(1.2, (pin_x, start_y, TOP + OVER_PANEL), STRADDLE, "over the panel"),
        Waypoint(1.2, (pin_x, start_y, grip_z), STRADDLE, "fingers either side of the pin"),
        Waypoint(2.0, (pin_x, upright_y, grip_z), STRADDLE, "push the pin across"),
    ]
    t_turned = sum(w.duration for w in right)
    left = [
        Waypoint(1.5, (GRIP_X, 0.12, TOP + 0.13), FINGER_OPEN, "towards the handle"),
        Waypoint(1.0, (GRIP_X, 0.0, TOP + 0.13), FINGER_OPEN, "above the handle"),
        Waypoint(1.0, (GRIP_X, 0.0, z), FINGER_OPEN, "down to the handle"),
        Waypoint(0.8, (GRIP_X, 0.0, z), FINGER_CLOSED, "grip"),
    ]
    failure = os.environ.get("TASK_DEMO") == "failure"
    if failure:
        left += [
            Waypoint(2.0, (GRIP_X - PULL_TO_STOP, 0.0, z), FINGER_CLOSED, "pull"),
            Waypoint(2.0, (GRIP_X - PULL_TO_STOP, 0.0, z), FINGER_CLOSED, "pull harder"),
            Waypoint(0.6, (GRIP_X - PULL_TO_STOP, 0.0, z), FINGER_OPEN, "let go"),
            Waypoint(1.0, (GRIP_X - PULL_TO_STOP, 0.0, TOP + 0.13), FINGER_OPEN, "rise"),
            Waypoint(1.2, LEFT_CLEAR, FINGER_OPEN, "clear"),
            Waypoint(3.0, LEFT_CLEAR, FINGER_OPEN, "watch"),
        ]
        return {"left": left}
    wait = t_turned + 0.3 - sum(w.duration for w in left)
    left.append(Waypoint(max(wait, 0.1), (GRIP_X, 0.0, z), FINGER_CLOSED, "wait for the head"))
    left.append(Waypoint(1.5, (GRIP_X - PULL_INTO_SLOT, 0.0, z), FINGER_CLOSED, "pull the head into the slot"))
    t_slot = sum(w.duration for w in left)
    left.append(Waypoint(2.4, (GRIP_X - PULL_INTO_SLOT, 0.0, z), FINGER_CLOSED, "hold while the other arm leaves"))
    left += [
        Waypoint(1.5, (GRIP_X - PULL_TO_STOP, 0.0, z), FINGER_CLOSED, "pull to the stop"),
        Waypoint(0.5, (GRIP_X - PULL_TO_STOP, 0.0, z), FINGER_CLOSED, "hold"),
        Waypoint(0.6, (GRIP_X - PULL_TO_STOP, 0.0, z), FINGER_OPEN, "let go"),
        Waypoint(1.0, (GRIP_X - PULL_TO_STOP, 0.0, TOP + 0.13), FINGER_OPEN, "rise"),
        Waypoint(1.2, LEFT_CLEAR, FINGER_OPEN, "clear"),
        Waypoint(3.0, LEFT_CLEAR, FINGER_OPEN, "watch"),
    ]
    right += [
        Waypoint(max(t_slot - t_turned + 0.2, 0.1), (pin_x, upright_y, grip_z), STRADDLE, "hold it upright"),
        Waypoint(0.8, (pin_x, upright_y, TOP + 0.14), STRADDLE, "lift off the pin"),
        Waypoint(1.0, (0.24, -0.12, TOP + OVER_PANEL + 0.01), STRADDLE, "back over the panel"),
        Waypoint(1.2, RIGHT_CLEAR, FINGER_OPEN, "clear"),
    ]
    return {"left": left, "right": right}
