"""On the station table (an oak-effect laminate top): 20 toasted oat cereal rings, each 12 mm across, are scattered over a
20 x 20 cm patch in front of the right arm, just right of the arms' midline. A blue plastic hand dustpan (25 cm wide) lies
flat on the table on the left, its black rubber lip facing the spill 6 cm from it and its handle pointing back towards the
left arm. A wooden hand brush (28 cm) lies behind the spill on the right arm's side, its bristles down and its handle
pointing towards the arms.

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

# -- layout (world frame: x forward from the arms, y left, metres) -----------------------------------------------------

PAN_XY = (0.34, 0.0)        # the middle of the lip's front edge; the pan opens towards -y
BRUSH_XY = (0.35, -0.30)    # the middle of the bristles; the brush lies along x, its handle towards the arms
SPILL_XY = (0.35, -0.16)    # centre of the 20 x 20 cm patch
RING_XY = [  # ring centres relative to SPILL_XY (a fixed random scatter, at least 24 mm apart)
    (-0.025, -0.093), (+0.021, -0.086), (-0.092, -0.058), (-0.054, -0.064), (-0.001, -0.047),
    (+0.052, -0.052), (-0.037, -0.042), (-0.083, -0.021), (-0.046, -0.010), (+0.056, -0.006),
    (-0.087, +0.003), (+0.024, +0.003), (+0.001, +0.010), (-0.026, +0.018), (+0.026, +0.045),
    (-0.093, +0.060), (+0.093, +0.055), (-0.038, +0.070), (+0.024, +0.075), (-0.006, +0.078),
]
RINGS = [f"ring{i:02d}" for i in range(len(RING_XY))]

DUSTPAN = UsdBody("dustpan", usd="assets/dustpan.usda", pos=(*PAN_XY, TOP + 0.0003), size=(0.25, 0.335, 0.075),
                  mass=sa.PAN_MASS, friction=None, solver_iters=32, torsional_patch=0.008)
BRUSH = UsdBody("brush", usd="assets/brush.usda", pos=(*BRUSH_XY, TOP + 0.0003), size=(0.284, 0.046, 0.042),
                mass=sa.BRUSH_MASS, friction=None, solver_iters=32, torsional_patch=0.008)
_ring_specs = []
for i, (dx, dy) in enumerate(RING_XY):
    yaw = 0.7 * i
    _ring_specs.append(UsdBody(RINGS[i], usd="assets/ring.usda",
                               pos=(SPILL_XY[0] + dx, SPILL_XY[1] + dy, TOP + sa.RING_HALF_T + 0.0003),
                               rot_xyzw=(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)), size=(0.012, 0.012, 0.005),
                               mass=sa.RING_MASS, friction=None, solver_iters=16))
OBJECTS = [DUSTPAN, BRUSH, *_ring_specs]

# -- what the scorer measures ------------------------------------------------------------------------------------------

FELL_BELOW = 0.10           # a tool whose origin drops this far below the table top has fallen off it
PAST_LIP = sa.LIP_D         # a ring's centre at least this far behind the lip's front edge is past the lip (3 cm)
IN_PAN_LOW = sa.PAN_FLOOR + sa.RING_HALF_T - 0.001  # ... its centre on the pan's floor (a ring flat on it: 4.5 mm up),
IN_PAN_HEIGHT = 0.03        # ... and at most this far above the pan's bottom: it rests in the pan, not under it
FLAT_TILT = math.radians(5.0)  # the pan is flat: tilted less than this ...
FLAT_LIP_GAP = 0.005        # ... with both ends of its lip within this of the table
TOOL_CLEAR = 0.02           # let go: every fingertip at least this far from both tools
RING_CLEAR = 0.05           # every fingertip at least this far from every ring
STILL_MOVE = 0.005          # no ring moving more than this ...
STILL_STEPS = 20            # ... over the last 1 s at 20 Hz (21 samples)
TIP_IN_GRIPPER = np.array([0.0, 0.0, TCP_OFFSET[2] + PAD_TIP_PAST_TCP])

_PAN_BOXES = [(p.min(axis=0), p.max(axis=0)) for _, p, _, _ in sa.pan_colliders()]
_BRUSH_BOXES = [(p.min(axis=0), p.max(axis=0)) for _, p, _ in sa.brush_colliders()]
_LIP_ENDS = np.array([[-sa.PAN_W / 2, 0.0, 0.0], [sa.PAN_W / 2, 0.0, 0.0]])

_S: dict = {}


def _clear() -> None:
    _S.clear()
    _S.update(step=0, fell=False, tips=None, history=[], cache=None)


_clear()


def _np(x) -> np.ndarray:
    return getattr(x, "torch", x).detach().cpu().numpy().astype(np.float64)


def _rotation(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _fingertips(env) -> np.ndarray:
    """The middle of each finger pad's tip, four points, in the table's frame."""
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


def _ring_positions(env) -> np.ndarray:
    base = env.unwrapped
    origin = _np(base.scene.env_origins[0])
    return np.array([_np(base.scene[n].data.root_link_pose_w)[0, :3] - origin for n in RINGS])


@on_reset
def _reset(env) -> None:
    """Failures stay failed until a real reset."""
    _clear()


@on_step
def _step(env) -> None:
    _S["step"] += 1
    _S["tips"] = _fingertips(env)
    _S["history"].append(_ring_positions(env))
    del _S["history"][:-(STILL_STEPS + 1)]
    if os.environ.get("TASK_DEBUG") and os.environ.get("TASK_SCORE_DIR"):
        _log(env)


def _log(env) -> None:
    """Debug telemetry (TASK_DEBUG=1): poses of the tools, the rings and the fingertips every control step."""
    import json

    base = env.unwrapped
    origin = _np(base.scene.env_origins[0])
    row = {"t": _S["step"] / 20.0, "tips": np.round(_S["tips"], 5).tolist(),
           "rings": np.round(_S["history"][-1], 5).tolist()}
    for name in ("dustpan", "brush"):
        pose = _np(base.scene[name].data.root_link_pose_w)[0]
        row[name] = np.round(np.r_[pose[:3] - origin, pose[3:7]], 5).tolist()
    folder = pathlib.Path(os.environ["TASK_SCORE_DIR"])
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "telemetry.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def _to_boxes(points: np.ndarray, pos: np.ndarray, rot: np.ndarray, boxes) -> float:
    """The smallest distance from the points to a body's boxes (bounds in the body's frame)."""
    local = (points - pos) @ rot
    best = np.inf
    for lo, hi in boxes:
        d = np.linalg.norm(np.maximum(np.maximum(lo - local, local - hi), 0.0), axis=1)
        best = min(best, float(d.min()))
    return best


def _evaluate(objects) -> dict:
    """Everything the score needs, once per control step."""
    if _S["cache"] is not None and _S["cache"][0] == _S["step"]:
        return _S["cache"][1]
    pan, brush = objects["dustpan"], objects["brush"]
    pan_rot, brush_rot = pan.rotation(), brush.rotation()
    if pan.pos[2] < TOP - FELL_BELOW or brush.pos[2] < TOP - FELL_BELOW:
        _S["fell"] = True
    rings = np.array([objects[n].pos for n in RINGS])
    local = (rings - pan.pos) @ pan_rot
    in_pan = ((local[:, 1] >= PAST_LIP) & (local[:, 1] <= sa.INNER_BACK) & (np.abs(local[:, 0]) <= sa.INNER_HALF_W)
              & (local[:, 2] >= IN_PAN_LOW) & (local[:, 2] <= IN_PAN_HEIGHT))
    tilt = math.acos(max(-1.0, min(1.0, float(pan_rot[2, 2]))))
    lip_z = (pan.pos + _LIP_ENDS @ pan_rot.T)[:, 2] - TOP
    flat = tilt < FLAT_TILT and bool(np.all(np.abs(lip_z) < FLAT_LIP_GAP))
    tips = _S["tips"]
    if tips is None:
        released, clear = False, False
        tool_gap = ring_gap = 0.0
    else:
        tool_gap = min(_to_boxes(tips, pan.pos, pan_rot, _PAN_BOXES), _to_boxes(tips, brush.pos, brush_rot, _BRUSH_BOXES))
        ring_gap = float(np.linalg.norm(tips[:, None, :] - rings[None, :, :], axis=2).min())
        released, clear = tool_gap >= TOOL_CLEAR, ring_gap >= RING_CLEAR
    hist = _S["history"]
    if len(hist) >= STILL_STEPS + 1:
        h = np.array(hist)  # [steps, rings, 3]
        spread = np.linalg.norm(h[:, None, :, :] - h[None, :, :, :], axis=3).max()
        still = bool(spread <= STILL_MOVE)
    else:
        spread, still = np.inf, False
    out = dict(n_in=int(in_pan.sum()), in_pan=in_pan, flat=flat, tilt_deg=math.degrees(tilt), lip_z=lip_z,
               released=released, clear=clear, tool_gap=tool_gap, ring_gap=ring_gap, still=still, spread=spread,
               fell=_S["fell"])
    _S["cache"] = (_S["step"], out)
    return out


def _partial(n_in: int) -> float:
    if n_in >= 18:
        return 0.75
    if n_in >= 10:
        return 0.5
    if n_in >= 5:
        return 0.25
    return 0.0


def score(objects) -> float:
    """0 if the dustpan or the brush ever falls off the table. Otherwise by the rings in the pan: 0.25 for at least 5, 0.5
    for at least 10, 0.75 for at least 18. 1.0 comes only from success(), held for 1 s."""
    e = _evaluate(objects)
    if e["fell"]:
        return 0.0
    return _partial(e["n_in"])


def success(objects) -> bool:
    """All 20 rings in the pan, the pan flat on the table, every fingertip clear of both tools and of the rings, and no
    ring moving more than 5 mm over the last second. The harness then needs it for 1 s in a row."""
    steptime.tick()  # the harness calls success() once per control step
    e = _evaluate(objects)
    ok = (not e["fell"] and e["n_in"] == len(RINGS) and e["flat"] and e["released"] and e["clear"] and e["still"])
    if os.environ.get("TASK_DEBUG") and _S["step"] % 20 == 0:
        print(f"[task] t={_S['step'] / 20:.1f} in={e['n_in']} flat={e['flat']} tilt={e['tilt_deg']:.2f} "
              f"lip={np.round(e['lip_z'] * 1000, 1)} tool_gap={e['tool_gap']:.3f} ring_gap={e['ring_gap']:.3f} "
              f"spread={e['spread']:.4f} fell={e['fell']}", flush=True)
    return ok


# -- the scripted demo -------------------------------------------------------------------------------------------------

LEFT_YAW = -math.pi / 2     # fingers close along x, across the pan's handle
HOVER = TOP + 0.14
PAN_GRIP = (PAN_XY[0], PAN_XY[1] + sa.GRIP_Y, TOP + sum(sa.HANDLE_Z) / 2)
BRUSH_GRIP_Z = TOP + sum(sa.HANDLE_ZB) / 2
PRESS = 0.0                 # strokes hold the brush at its resting height: pressing down turns it in the pinch
LIFT = 0.025                # the bristles clear the rings by this much between strokes
FAILURE_PRESS = 0.02        # the failure presses the handle this far down
HOLD_TILT = 0.004           # the demo lifts the handle this much once it holds it, tipping the pan onto its lip


def _brush_grip(xc: float, y: float, dz: float = 0.0) -> tuple[float, float, float]:
    """The right TCP for the brush's bristles centred at (xc, y), raised dz above resting on the table."""
    return (xc + sa.BRUSH_GRIP_X, y, BRUSH_GRIP_Z + dz)


SWEEP_OFFSETS = (-0.02, 0.02)  # the two sweep-in strokes, either side of the pan's centre line
SWEEP_FROM = -0.13          # where the sweep-in strokes start (y of the bristles' centre)
SWEEP_PAST_LIP = 0.03       # where they end: the bristles' centre this far past the lip's front edge
PLACE_AT_STEP = 160         # 8 s: the pan has settled in the left hand; the sweep-in strokes are placed from its pose

_PLAN: dict = {}


def _sweep_in(xc: float, lip_y: float) -> list[tuple[float, tuple[float, float, float], str]]:
    """One sweep-in stroke along x = xc for a lip at lip_y: (duration, TCP, label) per waypoint."""
    return [
        (1.5, _brush_grip(xc, lip_y + SWEEP_FROM, LIFT), "to the sweep"),
        (0.8, _brush_grip(xc, lip_y + SWEEP_FROM, -PRESS), "brush down"),
        (2.0, _brush_grip(xc, lip_y - 0.03, -PRESS), "sweep"),
        (2.5, _brush_grip(xc, lip_y + SWEEP_PAST_LIP, -PRESS + sa.PAN_FLOOR), "sweep over the lip"),
        (0.8, _brush_grip(xc, lip_y + SWEEP_PAST_LIP, LIFT + 0.01), "brush up"),
    ]


def _right_plan() -> tuple[list[Waypoint], float]:
    c, o = FINGER_CLOSED, FINGER_OPEN
    x0, y0 = BRUSH_XY
    w = [
        Waypoint(2.5, _brush_grip(x0, y0, 0.09), o, "to the brush"),
        Waypoint(1.5, _brush_grip(x0, y0), o, "descend"),
        Waypoint(1.0, _brush_grip(x0, y0), c, "grip the brush"),
        Waypoint(0.8, _brush_grip(x0, y0, 0.012), c, "lift"),
    ]
    # Gather: two strokes towards the pan, stopping short of the lip, over the left and right halves of the spill.
    for xc in (0.32, 0.38):
        w += [
            Waypoint(1.5, _brush_grip(xc, y0, LIFT), c, "to the stroke"),
            Waypoint(0.8, _brush_grip(xc, y0, -PRESS), c, "brush down"),
            Waypoint(4.5, _brush_grip(xc, -0.045, -PRESS), c, "gather"),
            Waypoint(0.8, _brush_grip(xc, -0.045, LIFT), c, "brush up"),
        ]
    # Sweep in: two strokes from behind the gathered rings, over the lip and 3 cm into the pan, kept inside its walls.
    # Planned here for the pan where it lies; _place_sweeps() moves them to the pan's measured pose once it is held.
    sweeps = []
    for dx in SWEEP_OFFSETS:
        for dur, pos, label in _sweep_in(PAN_XY[0] + dx, PAN_XY[1]):
            sweeps.append(Waypoint(dur, pos, c, label))
    w += sweeps
    _PLAN.update(sweeps=sweeps, placed=False)
    swept = sum(p.duration for p in w)
    w += [
        Waypoint(2.0, _brush_grip(x0, y0, LIFT + 0.01), c, "carry the brush back"),
        Waypoint(1.0, _brush_grip(x0, y0, 0.001), c, "set the brush down"),
        Waypoint(0.8, _brush_grip(x0, y0, 0.001), o, "let go"),
        Waypoint(1.5, _brush_grip(x0, y0, 0.09), o, "clear"),
    ]
    return w, swept


@on_step
def _place_sweeps(env) -> None:
    """The scripted demo reads where the held dustpan's lip is (what the top camera shows) and aims its sweep-in strokes
    at it. Only the demo's waypoints change; the simulation is not touched."""
    if not _PLAN.get("sweeps") or _PLAN.get("placed") or _S["step"] < PLACE_AT_STEP:
        return
    base = env.unwrapped
    pose = _np(base.scene["dustpan"].data.root_link_pose_w)[0]
    origin = _np(base.scene.env_origins[0])
    lip_x, lip_y = (pose[:3] - origin)[:2]
    planned = [s for dx in SWEEP_OFFSETS for s in _sweep_in(lip_x + dx, lip_y)]
    for wp, (_, pos, _) in zip(_PLAN["sweeps"], planned):
        wp.pos = pos
    _PLAN["placed"] = True
    if os.environ.get("TASK_DEBUG"):
        print(f"[task] sweep-in strokes placed for the lip at ({lip_x:.4f}, {lip_y:.4f})", flush=True)


def _left_plan(hold_until: float, tilted: bool) -> list[Waypoint]:
    c, o = FINGER_CLOSED, FINGER_OPEN
    x, y, z = PAN_GRIP
    w = [
        Waypoint(2.5, (x, y, HOVER), o, "to the handle", LEFT_YAW),
        Waypoint(1.5, (x, y, z), o, "descend", LEFT_YAW),
        Waypoint(1.0, (x, y, z), c, "grip the handle", LEFT_YAW),
    ]
    if tilted:  # the failure: the handle pressed down, so the pan tips back on its heel and its lip lifts
        hold_z = z - FAILURE_PRESS
        w.append(Waypoint(1.0, (x, y, hold_z), c, "press the handle down", LEFT_YAW))
    else:  # tip the pan forward onto its lip, as a hand does: the lip carries the pan's weight and stays down
        hold_z = z + HOLD_TILT
        w.append(Waypoint(1.0, (x, y, hold_z), c, "tip the pan onto its lip", LEFT_YAW))
    held = sum(p.duration for p in w)
    w.append(Waypoint(max(0.5, hold_until - held), (x, y, hold_z), c, "hold the pan", LEFT_YAW))
    if not tilted:
        w.append(Waypoint(1.0, (x, y, z), c, "set the pan flat", LEFT_YAW))
    w += [
        Waypoint(0.8, (x, y, hold_z if tilted else z), o, "let go", LEFT_YAW),
        Waypoint(1.5, (x, y, HOVER), o, "clear", LEFT_YAW),
    ]
    return w


def demo():
    """TASK_DEMO=failure presses the dustpan's handle 2 cm down while the right arm sweeps: the pan tips back on its
    heel, its lip lifts off the table and the rings pass under it."""
    right, swept = _right_plan()
    return {"left": _left_plan(swept + 0.5, os.environ.get("TASK_DEMO") == "failure"), "right": right}
