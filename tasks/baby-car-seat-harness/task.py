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

SEAT = UsdBody("seat", usd="assets/seat.usda", pos=(0.0, 0.0, TOP), size=(0.4, 0.3, 0.2), friction=(0.6, 0.5))
HARNESS = UsdBody("harness", usd="assets/harness.usda", pos=(0.0, 0.0, TOP), size=(0.2, 0.2, 0.1), articulation=True,
                  joint_pos=((".*", 0.0),), joint_groups=(("strap_.*|tongue_.*", 0.01), ("clip_slide", 3.0), ("clip_half_.*", 1.0)),
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

_S: dict = {}


def _clear() -> None:
    _S.clear()
    _S.update(head=False, seat=False, latched={"l": False, "r": False}, closed=False, clip_u=sa.CLIP_START_U, tips=None, history=[], depth={"l": 0.0, "r": 0.0}, lock_in={"l": 0, "r": 0}, ready=False)


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
        _S["ready"] = True
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


@on_reset
def _on_reset(env) -> None:
    _clear()
    _unlock(env.unwrapped.scene["harness"])
    _measure(env)


@on_step
def _on_step(env) -> None:
    _measure(env)


def _at_rest() -> bool:
    h = _S["history"]
    return len(h) > REST_STEPS and abs(h[-1] - h[0]) < REST_MOVE


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
    return 1.0 if _S["tip_to_seat"] >= CLEAR and _at_rest() else 0.75


def success(objects) -> bool:
    steptime.tick()  # the harness calls success() once per control step
    return score(objects) == 1.0


def distance(objects) -> float:
    return abs(_S["clip_u"] - sa.ARMPIT_U) if _S else 1.0


# -- the scripted demo -------------------------------------------------------------------------------------------------

LEFT_CLEAR = (0.14, 0.305, TOP + 0.16)
RIGHT_CLEAR = (0.14, -0.305, TOP + 0.16)
ACROSS = {1: -math.pi / 2, -1: math.pi / 2}  # the grip that closes the fingers along x, per arm
GRIP_ABOVE_BODY = 0.006  # TCP above the middle of the tongue's body, so the pads clear the buckle's top
TRAVEL = 0.165
APPROACH = 0.178
PARTLY_OPEN = -0.022  # wide enough for a tongue; fully open, a finger reaches the clip or the seat's rim


def _tongue_to_buckle(side: int) -> list[Waypoint]:
    yaw = ACROSS[side]
    t = sa.tongue_start(side)
    s = sa.buckle_slot(side)
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
        Waypoint(1.0, clear, FINGER_OPEN, "clear"),
    ]


def _clip(to_armpits: bool) -> list[Waypoint]:
    start = sa.clip_start()
    finish = sa.seat_point(sa.ARMPIT_U - 0.012, 0.0, sa.RAIL_W)
    squeeze = FINGER_OPEN * 0.0
    x, z = start[0], start[2]
    moves = [
        Waypoint(1.5, (x, 0.0, TOP + TRAVEL), FINGER_OPEN, "above the clip"),
        Waypoint(1.2, (x, 0.0, TOP + z), FINGER_OPEN, "down to the clip"),
        Waypoint(1.2, (x, 0.0, TOP + z), squeeze, "squeeze it shut"),
    ]
    end_x, end_z = x, z
    if to_armpits:
        moves.append(Waypoint(2.5, (finish[0], 0.0, TOP + finish[2]), squeeze, "slide it up"))
        moves.append(Waypoint(0.4, (finish[0], 0.0, TOP + finish[2]), squeeze, "hold"))
        end_x, end_z = finish[0], finish[2]
    moves += [
        Waypoint(0.8, (end_x, 0.0, TOP + end_z), FINGER_OPEN, "let go"),
        Waypoint(1.2, (end_x, 0.0, TOP + 0.185), FINGER_OPEN, "rise"),
        Waypoint(1.5, RIGHT_CLEAR, FINGER_OPEN, "clear"),
        Waypoint(4.0, RIGHT_CLEAR, FINGER_OPEN, "watch"),
    ]
    return moves


def demo() -> dict[str, list[Waypoint]]:
    """TASK_DEMO=failure leaves the clip at the belly. Open loop, from the starting layout only."""
    to_armpits = os.environ.get("TASK_DEMO") != "failure"
    left = _tongue_to_buckle(1)
    wait = sum(w.duration for w in left)
    right = [Waypoint(wait, RIGHT_CLEAR, FINGER_OPEN, "wait"), *_tongue_to_buckle(-1), *_clip(to_armpits)]
    return {"left": left, "right": right}
