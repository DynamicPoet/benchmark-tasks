"""On the station table: a 500 mL narrow-neck Erlenmeyer flask with a red PTFE stir bar lying on its floor, an empty 250 mL
beaker beside it, and a magnetic stir-bar positioner/retriever (a 305 mm polyethylene tube with a magnet rod sliding
inside it, a ring handle on top) lying flat in front of them.

The robot takes the retriever, steers its tube down through the flask's neck to the bar, and the magnet picks the bar up
against the side of the tube. Tube and bar together are about 24 mm across and the neck's bore about 29 mm, so the tool
has to come up straight and centred: if the bar meets the neck, the glass scrapes it off and it falls back. Over the
beaker the tube is held still while the magnet rod is slid up by its ring, and the bar drops off.

The magnet (an approximation, not a field): while the rod is down and the bar comes within 6 mm of the tube's lower 40 mm
with no glass between them, the bar is placed lengthwise against the tube's side (its lower end at the tip) and held
there by a fixed joint; it is lost as soon as any part of it reaches the neck's glass or is pushed 3 mm off its place, and
let go when the rod slides up 12 mm. A bar lost or let go is picked up again only once it is 1 cm from the tube.

Run it:  inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \\
           -E control_hz=20 --scorer task_score --max-steps 3000 --instruction "<the task.json instruction>"
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # lets this file import the helpers beside it

import poseplay  # noqa: E402,F401  (lets demo() tilt the grippers)
import scene_assets as sa  # noqa: E402
import scenekit  # noqa: E402
import steptime  # noqa: E402
from poseplay import Joints, Pose  # noqa: E402
from scenekit import UsdBody, on_reset, on_step  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, READY, TCP_OFFSET  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 0.0  # score() holds the end state for 1 s itself
SCORE_OVER = "end"
INSTRUCTION = ("Take the stir bar out of the flask with the magnetic retriever and drop it into the beaker: hold the "
               "tube and slide its magnet up by the ring to let the bar go, then lift the retriever out of the beaker. "
               "Do not tip, pour or move the flask.")

sa.write_all()

# -- layout (world frame: x forward from the arms, y left, metres; heights above the table top) -------------------------

FLASK_XY = (0.30, -0.08)
BEAKER_XY = (0.30, 0.03)
TOOL_XY = (0.16, -0.05)          # the tube's middle; it lies along y, tip toward -y
TOOL_TIP = (TOOL_XY[0], TOOL_XY[1] - sa.TUBE_L / 2, TOP + sa.TUBE_R)
TOOL_Q = (-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))   # the tool's +z (tip to ring) along world +y
BAR_Z = TOP + sa.FLOOR + sa.BAR_R + 0.0003

FLASK = UsdBody("flask", usd="assets/flask.usda", pos=(*FLASK_XY, TOP), size=(0.105, 0.105, 0.18), mass=sa.FLASK_MASS,
                friction=None, solver_iters=32)
BEAKER = UsdBody("beaker", usd="assets/beaker.usda", pos=(*BEAKER_XY, TOP), size=(0.07, 0.07, 0.095), mass=sa.BEAKER_MASS,
                 friction=None, solver_iters=32)
BAR = UsdBody("bar", usd="assets/bar.usda", pos=(*FLASK_XY, BAR_Z), size=(sa.BAR_L, 0.008, 0.008), mass=sa.BAR_MASS,
              friction=None, solver_iters=32)
TOOL = UsdBody("tool", usd="assets/tool.usda", pos=TOOL_TIP, rot_xyzw=TOOL_Q, size=(0.022, 0.022, 0.34), articulation=True,
               joint_pos=(("slide", 0.0),), joint_damping=0.05, friction=None, solver_iters=32, torsional_patch=0.012)
OBJECTS = [FLASK, BEAKER, BAR, TOOL]

# -- the magnet ---------------------------------------------------------------------------------------------------------

PICK_RANGE = 0.006       # the bar is drawn to the tube from this close
MAGNET_ZONE = 0.040      # along the tube's lowest 40 mm
ROD_DOWN = 0.008         # the rod counts as down (magnet at the tip) below this slide
ROD_RELEASE = 0.012      # slid up this far, the magnet lets the bar go
HOLD_N = 1.0e3           # the joint itself does not give way: the bar is lost where it meets the neck's glass (below)
SIDE_GAP = 0.0005
SNAP_DRIFT = 0.003       # the bar more than this off its place on the tube: the hold has given way
BAR_CORNER = sa.BAR_R / math.cos(math.pi / 8)   # the octagon bar's corners: 4.33 mm from its axis
REARM = 0.004            # after it is lost or let go, the bar picks up again only once it has been this far beyond range
_R_SIDE = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])   # the bar's x along the tube's z

# -- what the scorer measures ------------------------------------------------------------------------------------------

TILT_MAX = math.radians(30.0)
CIRCLE = 0.02            # the flask's base may move this far on its taped circle
LIFT = 0.03              # 0.5: the bar's lowest point this far above the flask's floor
OUT = 0.01               # 0.75: ... this far above the rim
CLEAR = 0.03             # every fingertip this far from the bar at the end
REST_SPEED = 0.01
HOLD_STEPS = 21          # 1 s at 20 Hz

_S: dict = {}


def _clear() -> None:
    _S.clear()
    _S.update(step=0, armed=True, attached=False, joint_n=0, joint=None, p_local=None, release_slide=False, strip=0, picks=0,
              upset=False, tip_in=False, lifted=False, out=False, hold=0, last=None)


_clear()


# -- reading the simulator ---------------------------------------------------------------------------------------------


def _np(x) -> np.ndarray:
    return getattr(x, "torch", x).detach().cpu().numpy().astype(np.float64)


def _rotation(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """(x, y, z, w) of a rotation matrix."""
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return (x / n, y / n, z / n, w / n)


def _origin(env) -> np.ndarray:
    return _np(env.unwrapped.scene.env_origins[0])


def _pose(env, name: str) -> tuple[np.ndarray, np.ndarray]:
    p = _np(env.unwrapped.scene[name].data.root_link_pose_w)[0]
    return p[:3] - _origin(env), _rotation(p[3:7])


def _tool(env) -> tuple[np.ndarray, np.ndarray, float]:
    """The tube's tip, its rotation (z along the tube, tip to ring) and the rod's slide (m)."""
    asset = env.unwrapped.scene["tool"]
    poses = _np(asset.data.body_link_pose_w)[0]
    p = poses[asset.body_names.index("tube")]
    q = float(_np(asset.data.joint_pos)[0, asset.joint_names.index("slide")])
    return p[:3] - _origin(env), _rotation(p[3:7]), q


def _gripper(env, side: str) -> tuple[np.ndarray, np.ndarray, float]:
    base = env.unwrapped
    robot = base.scene[f"{side}_robot"]
    i = robot.find_bodies("gripper")[0][0]
    pose = _np(robot.data.body_link_pose_w)[0, i]
    opening = abs(float(_np(robot.data.joint_pos)[0, robot.find_joints("joint7")[0][0]]))
    return pose[:3] - _origin(env), _rotation(pose[3:7]), opening


def _fingertips(env) -> np.ndarray:
    """24 points on both hands' fingers (as kit/mechkit.fingertips): tip and 3 cm up, middle and both edges, each finger."""
    out = []
    for side in ("left", "right"):
        pos, R, opening = _gripper(env, side)
        for sx in (-1.0, 1.0):
            for dy in (-0.018, 0.0, 0.018):
                for dz in (0.147, 0.117):
                    out.append(pos + R @ np.array([sx * (opening + 0.0045), dy, dz]))
    return np.array(out)


def _tcp(env, side: str) -> np.ndarray:
    pos, R, _ = _gripper(env, side)
    return pos + R @ np.array(TCP_OFFSET)


# -- geometry ----------------------------------------------------------------------------------------------------------


def seg_dist(p0, p1, q0, q1) -> float:
    """Shortest distance between segments p0-p1 and q0-q1."""
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a < 1e-12 and e < 1e-12:
        return float(np.linalg.norm(r))
    if e < 1e-12:  # q is a point
        s = float(np.clip(-(d1 @ r) / a, 0.0, 1.0))
        return float(np.linalg.norm(p0 + s * d1 - q0))
    if a < 1e-12:  # p is a point
        t = float(np.clip(f / e, 0.0, 1.0))
        return float(np.linalg.norm(p0 - (q0 + t * d2)))
    c, b = d1 @ r, d1 @ d2
    den = a * e - b * b
    s = float(np.clip((b * f - c * e) / den, 0.0, 1.0)) if den > 1e-12 else 0.0
    t = (b * s + f) / e
    if t < 0.0:
        t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0))
    elif t > 1.0:
        t, s = 1.0, float(np.clip((b - c) / a, 0.0, 1.0))
    return float(np.linalg.norm((p0 + s * d1) - (q0 + t * d2)))


def bar_ends(pos: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    half = (sa.BAR_L / 2 - sa.BAR_R) * R[:, 0]
    return pos - half, pos + half


def bar_low(pos: np.ndarray, R: np.ndarray) -> float:
    """The bar's lowest point."""
    return float(pos[2] - abs(R[2, 0]) * (sa.BAR_L / 2 - sa.BAR_R) - sa.BAR_R)


def side_pose(tip: np.ndarray, Rt: np.ndarray, bar_pos: np.ndarray) -> np.ndarray:
    """Where on the tube the magnet draws the bar: lengthwise against the side facing it, its lower end at the tip."""
    local = Rt.T @ (bar_pos - tip)
    d = np.array([local[0], local[1]])
    n = float(np.linalg.norm(d))
    d = np.array([1.0, 0.0]) if n < 1e-4 else d / n
    r = sa.TUBE_R + sa.BAR_R + SIDE_GAP
    return np.array([r * d[0], r * d[1], sa.BAR_L / 2])


def local(p, frame) -> np.ndarray:
    """A point in a vessel's own frame (origin under the middle of its base, z up its axis), from a (pos, R) frame."""
    pos, R = frame
    return R.T @ (np.asarray(p, float) - pos)


def flask_inner_r(z: float) -> float:
    """The flask's inside radius at height z above its base (the flats of its glass)."""
    if z >= sa.SHOULDER_Z:
        return sa.NECK_BORE
    return max(sa._cone_r(max(z, sa.HEEL_Z)) - sa.WALL, sa.NECK_BORE)


def inside_flask(p, flask, margin: float = 0.0) -> bool:
    """A point inside the flask's glass, in the flask's own frame (so a leaning flask counts as it really is); `margin`
    widens it by a radius (for the tube's surface)."""
    q = local(p, flask)
    if q[2] < 0.0 or q[2] > sa.FLASK_H:
        return False
    return bool(math.hypot(q[0], q[1]) < flask_inner_r(float(q[2])) + margin)


def in_beaker_space(p, beaker, margin: float = 0.0) -> bool:
    q = local(p, beaker)
    return bool(math.hypot(q[0], q[1]) < sa.BEAKER_R + margin and 0.0 <= q[2] < sa.BEAKER_H)


def tube_in_vessels(tip, Rt, flask, beaker) -> bool:
    """Any part of the tube (its axis every 5 mm, widened by its radius) inside the flask or the beaker."""
    for s in np.arange(0.0, sa.TUBE_L + 1e-9, 0.005):
        p = tip + Rt[:, 2] * s
        if inside_flask(p, flask, sa.TUBE_R) or in_beaker_space(p, beaker, sa.TUBE_R):
            return True
    return False


def touches_neck(bar_pos, R_bar, flask) -> bool:
    """Any part of the bar out to the neck's bore while it is level with the neck, in the flask's own frame: the glass
    scrapes it off."""
    a, b = (local(x, flask) for x in bar_ends(bar_pos, R_bar))
    lo, hi = min(a[2], b[2]) - BAR_CORNER, max(a[2], b[2]) + BAR_CORNER
    if not (hi > sa.SHOULDER_Z and lo < sa.FLASK_H + 0.002):
        return False
    ra, rb = math.hypot(a[0], a[1]), math.hypot(b[0], b[1])
    outer = max(ra, rb) + BAR_CORNER    # the bar's farthest point from the neck's axis (an octagon's corner)
    inner = min(ra, rb) - BAR_CORNER    # its nearest
    return bool(outer >= sa.NECK_BORE - 0.0003 and inner <= sa.NECK_R + 0.0015)   # it reaches the neck's glass


def bar_low_in(bar_pos, R_bar, frame) -> float:
    """The bar's lowest point above a vessel's base, along the vessel's axis."""
    a, b = (local(x, frame) for x in bar_ends(bar_pos, R_bar))
    return float(min(a[2], b[2]) - BAR_CORNER)


# -- the magnet's joint, made on the live stage ------------------------------------------------------------------------

_PATHS: dict = {}


def _prim_paths() -> tuple[str, str]:
    if not _PATHS:
        from isaaclab.sim.utils.stage import get_current_stage

        stage = get_current_stage()
        for prim in stage.Traverse():
            p = str(prim.GetPath())
            if p.endswith("/tool/tube") and "tube" not in _PATHS:
                _PATHS["tube"] = p
            if p.endswith("/bar") and "bar" not in _PATHS and "Looks" not in p and "visual" not in p:
                _PATHS["bar"] = p
    return _PATHS["tube"], _PATHS["bar"]


def _joint_on(p_local: np.ndarray) -> str:
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, UsdPhysics

    stage = get_current_stage()
    tube, bar = _prim_paths()
    _S["joint_n"] += 1
    path = f"{tube.rsplit('/tool/', 1)[0]}/magnet_{_S['joint_n']}"
    j = UsdPhysics.FixedJoint.Define(stage, path)
    j.CreateBody0Rel().SetTargets([tube])
    j.CreateBody1Rel().SetTargets([bar])
    x, y, z, w = _quat(_R_SIDE)
    j.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in p_local]))
    j.CreateLocalRot0Attr(Gf.Quatf(w, x, y, z))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateExcludeFromArticulationAttr(True)
    j.CreateBreakForceAttr(HOLD_N)
    j.CreateBreakTorqueAttr(HOLD_N * sa.BAR_L / 2)
    j.CreateJointEnabledAttr(True)
    return path


def _joint_off(path: str | None) -> None:
    if not path:
        return
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import UsdPhysics

    prim = get_current_stage().GetPrimAtPath(path)
    if prim:
        UsdPhysics.Joint(prim).GetJointEnabledAttr().Set(False)


def _attach(env, tip, Rt, bar_pos) -> None:
    """The magnet draws the bar against the tube's side (it snaps there in a few milliseconds) and holds it."""
    import torch

    p_local = side_pose(tip, Rt, bar_pos)
    p_w = tip + Rt @ p_local
    q = _quat(Rt @ _R_SIDE)
    bar = env.unwrapped.scene["bar"]
    o = _origin(env)
    pose = torch.tensor([[p_w[0] + o[0], p_w[1] + o[1], p_w[2] + o[2], *q]], dtype=torch.float32, device=bar.device)
    bar.write_root_pose_to_sim_index(root_pose=pose)
    bar.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), dtype=torch.float32, device=bar.device))
    _S["joint"] = _joint_on(p_local)
    _S["p_local"] = p_local
    _S["attached"] = True
    _S["picks"] += 1


def _detach(why: str, over_beaker: bool = False) -> None:
    _S["armed"] = False
    _joint_off(_S["joint"])
    _S["joint"] = None
    _S["attached"] = False
    _S["release_slide"] = why == "slide" and over_beaker   # let go by the slide, over the beaker's opening
    if why == "strip":
        _S["strip"] += 1


def over_beaker(bar_pos, beaker) -> bool:
    """The bar above the beaker's opening or inside it (in the beaker's own frame)."""
    q = local(bar_pos, beaker)
    return bool(math.hypot(q[0], q[1]) < sa.BEAKER_R - sa.BEAKER_WALL and q[2] > sa.BEAKER_FLOOR)


def _magnet(env, tip, Rt, q_rod, bar_pos, R_bar, flask, beaker) -> None:
    if _S["attached"]:
        expected = tip + Rt @ _S["p_local"]
        if q_rod > ROD_RELEASE:
            _detach("slide", over_beaker(bar_pos, beaker))
        elif touches_neck(bar_pos, R_bar, flask) or np.linalg.norm(bar_pos - expected) > SNAP_DRIFT:
            _detach("strip")
        return
    a, b = bar_ends(bar_pos, R_bar)
    zone = seg_dist(a, b, tip + Rt[:, 2] * sa.TUBE_R, tip + Rt[:, 2] * MAGNET_ZONE) - sa.BAR_R - sa.TUBE_R
    if not _S["armed"]:
        _S["armed"] = zone > PICK_RANGE + REARM   # a bar scraped off or let go must first fall clear of the magnet
        return
    if q_rod > ROD_DOWN:
        return
    if inside_flask(bar_pos, flask) != inside_flask(tip, flask):
        return   # the glass is between them: the magnet does not draw the bar through it
    if zone < PICK_RANGE:
        _attach(env, tip, Rt, bar_pos)


# -- tracking, every control step --------------------------------------------------------------------------------------


@on_reset
def _reset(env) -> None:
    """Failures stay failed until a real reset."""
    for n in range(1, _S.get("joint_n", 0) + 1):
        try:
            tube, _ = _prim_paths()
            _joint_off(f"{tube.rsplit('/tool/', 1)[0]}/magnet_{n}")
        except Exception:  # noqa: BLE001
            pass
    n = _S.get("joint_n", 0)
    _clear()
    _S["joint_n"] = n


def _tilt(R: np.ndarray) -> float:
    return math.acos(float(np.clip(R[2, 2], -1.0, 1.0)))


@on_step
def _step(env) -> None:
    S = _S
    S["step"] += 1
    tip, Rt, q_rod = _tool(env)
    bar_pos, R_bar = _pose(env, "bar")
    flask_pos, R_flask = _pose(env, "flask")
    beaker_pos, R_beaker = _pose(env, "beaker")
    flask, beaker = (flask_pos, R_flask), (beaker_pos, R_beaker)
    _magnet(env, tip, Rt, q_rod, bar_pos, R_bar, flask, beaker)
    bar_pos, R_bar = _pose(env, "bar")
    vel = _np(env.unwrapped.scene["bar"].data.root_vel_w)[0]
    # failures that stay failed
    if _tilt(R_flask) > TILT_MAX or np.linalg.norm(flask_pos[:2] - np.array(FLASK_XY)) > CIRCLE:
        S["upset"] = True
    # milestones
    if inside_flask(tip, flask):
        S["tip_in"] = True
    low = bar_low_in(bar_pos, R_bar, flask)
    if low >= sa.FLOOR + LIFT:
        S["lifted"] = True
    if low >= sa.FLASK_H + OUT:
        S["out"] = True
    if S["release_slide"] and not S["attached"] and not over_beaker(bar_pos, beaker):
        S["release_slide"] = False   # a bar that leaves the beaker after the release does not count back in
    # the end state
    bq = local(bar_pos, beaker)
    in_beaker = math.hypot(bq[0], bq[1]) < sa.BEAKER_R - sa.BEAKER_WALL and bq[2] < sa.BEAKER_FLOOR + 0.02
    tool_in = tube_in_vessels(tip, Rt, flask, beaker)
    a, b = bar_ends(bar_pos, R_bar)
    tube_gap = seg_dist(a, b, tip + Rt[:, 2] * sa.TUBE_R, tip + Rt[:, 2] * (sa.TUBE_L - sa.TUBE_R)) - BAR_CORNER - sa.TUBE_R
    tips = _fingertips(env)
    hand_gap = min(seg_dist(a, b, p, p) for p in tips) - BAR_CORNER
    end = end_state(in_beaker, S["attached"], S["release_slide"], tube_gap, tool_in, _tilt(R_flask), _tilt(R_beaker),
                    hand_gap, float(np.linalg.norm(vel[:3])))
    S["hold"] = S["hold"] + 1 if end else 0
    S["last"] = dict(tip=tip, q_rod=q_rod, bar=bar_pos, low=low, in_beaker=in_beaker, tool_in=tool_in,
                     tube_gap=tube_gap, hand_gap=hand_gap)
    if os.environ.get("TASK_SCORE_DIR"):
        _log(env)


def end_state(in_beaker: bool, attached: bool, release_slide: bool, tube_gap: float, tool_in: bool, flask_tilt: float,
              beaker_tilt: float, hand_gap: float, bar_speed: float) -> bool:
    """The 1.0 state, before its 1 s hold: the bar lying in the beaker, let go by the magnet slid up (not knocked or
    scraped off), the tube out of both vessels and 1 cm off the bar, every fingertip 3 cm off it, flask and beaker upright,
    the bar still."""
    return bool(in_beaker and not attached and release_slide and tube_gap >= 0.01 and not tool_in
                and flask_tilt < math.radians(10.0) and beaker_tilt < math.radians(10.0) and hand_gap >= CLEAR
                and bar_speed < REST_SPEED)


def evaluate(S: dict) -> float:
    """0: the flask upset (tilted past 30 degrees or off its circle) at any moment, or nothing reached. Otherwise the best
    reached: 0.25 the tube's tip inside the flask; 0.5 the bar lifted 3 cm off the floor; 0.75 the whole bar 1 cm above the
    rim; 1.0 the bar resting in the beaker, let go by the magnet slid up, the tube out of both vessels and 1 cm off the
    bar, every fingertip 3 cm off it, held for 1 s."""
    if S["upset"]:
        return 0.0
    if S["hold"] >= HOLD_STEPS:
        return 1.0
    if S["out"]:
        return 0.75
    if S["lifted"]:
        return 0.5
    if S["tip_in"]:
        return 0.25
    return 0.0


def score(objects) -> float:
    return evaluate(_S)


def success(objects) -> bool:
    steptime.tick()  # the harness calls success() once per control step
    return evaluate(_S) == 1.0


def _log(env) -> None:
    S = _S
    folder = pathlib.Path(os.environ["TASK_SCORE_DIR"])
    folder.mkdir(parents=True, exist_ok=True)
    L = S["last"]
    row = {"step": S["step"], "score": evaluate(S), "attached": S["attached"], "picks": S["picks"], "strip": S["strip"],
           "release_slide": S["release_slide"], "upset": S["upset"], "tip_in": S["tip_in"], "lifted": S["lifted"],
           "out": S["out"], "hold": S["hold"], "q_rod_mm": round(1000 * L["q_rod"], 2),
           "tip": [round(float(v), 4) for v in L["tip"]], "bar": [round(float(v), 4) for v in L["bar"]],
           "bar_low": round(L["low"], 4), "in_beaker": bool(L["in_beaker"]), "tool_in": bool(L["tool_in"]),
           "tube_gap": round(L["tube_gap"], 4), "hand_gap": round(L["hand_gap"], 4),
           "tcp_left": [round(float(v), 4) for v in _tcp(env, "left")],
           "tcp_right": [round(float(v), 4) for v in _tcp(env, "right")],
           "flask": [round(float(v), 4) for v in _pose(env, "flask")[0]],
           "axis": [round(float(v), 4) for v in _tool(env)[1][:, 2]]}
    with open(folder / "track.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


# -- the scripted demo -------------------------------------------------------------------------------------------------

MODE = os.environ.get("TASK_DEMO", "success")
GRIP = 0.21              # the right hand holds the tube this far above its tip
DEPTH = 0.006            # the tube's axis this far past the TCP, toward the fingertips
BUNDLE = (sa.TUBE_R + 2 * sa.BAR_R + SIDE_GAP - sa.TUBE_R) / 2   # tube plus bar: their middle is this far off the tube's axis


def _rot(f, jaw) -> np.ndarray:
    f = np.asarray(f, float) / np.linalg.norm(f)
    jaw = np.asarray(jaw, float) - np.dot(jaw, f) * f
    jaw /= np.linalg.norm(jaw)
    R = np.zeros((3, 3))
    R[:, 2], R[:, 0] = f, jaw
    R[:, 1] = np.cross(f, jaw)
    return R


DOWN_X = _rot((0, 0, -1), (1, 0, 0))     # fingers down, jaws along x: across the lying tube
LEVEL_Y = _rot((0, 1, 0), (1, 0, 0))     # fingers level toward +y, jaws along x: the tube upright between them
RING_GRIP = _rot((0, -1, 0), (-1, 0, 0))  # left hand, fingers level toward -y, jaws along x: across the ring's 22 mm
OPEN_TUBE = -0.022
OPEN_RING = -0.020


def _prev(plan) -> np.ndarray:
    return np.asarray(plan._hp["pos"], float)


def _shift(dx: float, dy: float, dz: float):
    def at(plan):
        return tuple(_prev(plan) + np.array([dx, dy, dz]))

    return at


def _tool_now():
    env = scenekit.env()
    if env is None:
        tip = np.array(TOOL_TIP)
        return tip, _rotation(TOOL_Q)
    tip, Rt, _ = _tool(env)
    return tip, Rt


def _grip_down(dz: float):
    """Right hand over the lying tube, GRIP above its tip."""
    def at(plan):
        tip, Rt = _tool_now()
        a = tip + Rt[:, 2] * GRIP
        return tuple(a - DOWN_X[:, 2] * DEPTH + np.array([0.0, 0.0, dz]))

    return at


def _tip_to(goal, gain: float = 1.0):
    """Move the right hand so that the tube's tip reaches goal(): the last command corrected by where the tip really is
    (by `gain` of the error, for short steps the arm has not finished)."""
    def at(plan):
        g = np.asarray(goal(), float)
        env = scenekit.env()
        if env is None:
            return tuple(g + LEVEL_Y[:, 2] * -DEPTH + np.array([0.0, 0.0, GRIP]))
        tip, _, _ = _tool(env)
        return tuple(_prev(plan) + gain * (g - tip))

    return at


def _bar_side() -> np.ndarray:
    """The unit vector, level, from the tube's axis to the bar (the side the magnet drew it to)."""
    env = scenekit.env()
    if env is None or _S.get("p_local") is None:
        return np.array([1.0, 0.0, 0.0])
    tip, Rt, _ = _tool(env)
    d = Rt @ np.array([_S["p_local"][0], _S["p_local"][1], 0.0])
    d[2] = 0.0
    return d / max(1e-6, np.linalg.norm(d))


def _bundle_shift(env) -> np.ndarray:
    """Where the tip must go (level offset from the neck's axis) so that tube and bar together, as they really sit, are
    centred: the middle between the tube's far side and the bar's far side."""
    tip, _, _ = _tool(env)
    bar, _ = _pose(env, "bar")
    s = np.array([bar[0] - tip[0], bar[1] - tip[1], 0.0])
    n = float(np.linalg.norm(s))
    if n < 1e-6 or not _S.get("attached"):
        return np.zeros(3)
    s /= n
    centre = 0.5 * ((tip - s * sa.TUBE_R) + (bar + s * sa.BAR_R))
    return np.array([tip[0] - centre[0], tip[1] - centre[1], 0.0])


def _flask_at(z: float, centred: bool = False):
    def goal():
        env = scenekit.env()
        c = np.array([*FLASK_XY, TOP]) if env is None else _pose(env, "flask")[0]
        if centred and env is not None:
            off = _bundle_shift(env)
        else:
            off = -BUNDLE * _bar_side() if centred else np.zeros(3)
        return np.array([c[0] + off[0], c[1] + off[1], c[2] + z])

    return goal


def _over_bar(z: float, dy: float = -0.004):
    """The tip over the bar on the floor, a little to its -y side (so the magnet draws it to the tube's +y side, where the
    arm holds the tube steadiest), coming straight down the neck; once the magnet holds the bar, stay where it is."""
    def goal():
        env = scenekit.env()
        if env is None:
            return np.array([FLASK_XY[0], FLASK_XY[1] + dy, TOP + sa.FLOOR + 2 * sa.BAR_R + z])
        tip, _, _ = _tool(env)
        if _S.get("attached"):
            return tip
        bar, _ = _pose(env, "bar")
        flask, _ = _pose(env, "flask")
        return np.array([bar[0], bar[1] + dy, flask[2] + sa.FLOOR + 2 * sa.BAR_R + z])

    return goal


def _beaker_at(z: float):
    def goal():
        env = scenekit.env()
        c = np.array([*BEAKER_XY, TOP]) if env is None else _pose(env, "beaker")[0]
        off = -BUNDLE * _bar_side()
        return np.array([c[0] + off[0], c[1] + off[1], c[2] + z])

    return goal


def _ring_target(dz: float, dy: float) -> np.ndarray:
    tip, Rt = _tool_now()
    if scenekit.env() is None:
        tip, Rt = np.array([BEAKER_XY[0], BEAKER_XY[1], TOP + 0.045]), np.eye(3)
    r = tip + Rt[:, 2] * sa.RING_Z
    return r - RING_GRIP[:, 2] * 0.004 + np.array([0.0, dy, dz])


def _ring(dz: float, dy: float = 0.0):
    """Left hand at the ring handle on top of the tube (read from where the tool is)."""
    def at(plan):
        return tuple(_ring_target(dz, dy))

    return at


def _ring_fix(dz: float, dy: float = 0.0):
    """The same, the last command corrected by where the left hand's TCP really is (the soft arm sags)."""
    def at(plan):
        g = _ring_target(dz, dy)
        env = scenekit.env()
        if env is None:
            return tuple(g)
        return tuple(_prev(plan) + (g - _tcp(env, "left")))

    return at


RIM = sa.FLASK_H
T_RIGHT_READY = 0.0


def _align(a, b) -> np.ndarray:
    """The rotation that turns unit vector a onto unit vector b."""
    a, b = np.asarray(a, float) / np.linalg.norm(a), np.asarray(b, float) / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    k = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + k + k @ k / (1.0 + c)


def _upright(plan):
    """The last commanded grip, turned so that the tube (as it really sits in the fingers) stands vertical."""
    env = scenekit.env()
    if env is None:
        return LEVEL_Y
    _, Rt, _ = _tool(env)
    a = Rt[:, 2]
    ang = math.acos(float(np.clip(a[2], -1.0, 1.0)))
    if ang > math.radians(6.0):  # never more than 6 degrees at once
        h = np.array([a[0], a[1], 0.0]) / max(1e-9, math.hypot(a[0], a[1]))
        a = math.cos(math.radians(6.0)) * np.array([0.0, 0.0, 1.0]) + math.sin(math.radians(6.0)) * h
    return _align(a, np.array([0.0, 0.0, 1.0])) @ np.asarray(plan._hp["rot"], float)


def _keep(plan):
    return LEVEL_Y if scenekit.env() is None else np.asarray(plan._hp["rot"], float)


def _right() -> list:
    c = MODE != "strip"
    floor = sa.FLOOR + 2 * sa.BAR_R
    hold = lambda d, label: Pose(d, _shift(0.0, 0.0, 0.0), FINGER_CLOSED, label, rot=_keep)  # noqa: E731
    fix = lambda goal, label, d=0.8, k=1.0: Pose(d, _tip_to(goal, k), FINGER_CLOSED, label, rot=_keep)  # noqa: E731
    w = [
        Pose(2.0, _grip_down(0.07), OPEN_TUBE, "over the retriever", rot=DOWN_X),
        Pose(1.5, _grip_down(0.0), OPEN_TUBE, "jaws round the tube", rot=DOWN_X),
        Pose(0.8, _grip_down(0.0), FINGER_CLOSED, "grip the tube", rot=DOWN_X),
        Pose(1.5, _shift(0.0, 0.0, 0.10), FINGER_CLOSED, "lift it", rot=DOWN_X),
        Pose(3.0, _shift(0.06, -0.01, 0.18), FINGER_CLOSED, "turn it upright", rot=LEVEL_Y),
        hold(1.0, "let it settle"),
        Pose(2.5, _tip_to(_flask_at(RIM + 0.05)), FINGER_CLOSED, "tip over the neck", rot=_keep),
        hold(1.0, "let it settle"),
        Pose(0.8, _shift(0.0, 0.0, 0.0), FINGER_CLOSED, "square the tube upright", rot=_upright),
        hold(0.6, "let it settle"),
        Pose(0.8, _shift(0.0, 0.0, 0.0), FINGER_CLOSED, "square the tube upright", rot=_upright),
        hold(0.6, "let it settle"),
        fix(_flask_at(RIM + 0.03), "line up over the neck"),
        hold(0.5, "let it settle"),
        fix(_flask_at(RIM + 0.03), "line up over the neck"),
        hold(0.5, "let it settle"),
        fix(_flask_at(RIM + 0.03), "line up over the neck"),
        fix(_flask_at(RIM - 0.04), "into the neck", 1.5),
        fix(_over_bar(0.03), "down into the flask", 1.5),
        fix(_over_bar(0.012), "down over the bar", 1.0),
        fix(_over_bar(0.004), "down to the bar", 1.0),
        fix(_over_bar(0.002), "down to the bar", 0.8),
        fix(_flask_at(sa.FLOOR + 0.02), "lift off the floor"),
        hold(0.5, "let it settle"),
        Pose(0.8, _shift(0.0, 0.0, 0.0), FINGER_CLOSED, "square the tube upright", rot=_upright),
        hold(0.5, "let it settle"),
        fix(_flask_at(sa.FLOOR + 0.02, centred=c), "centre tube and bar", 1.0),
        hold(0.5, "let it settle"),
        fix(_flask_at(sa.FLOOR + 0.02, centred=c), "centre tube and bar"),
        fix(_flask_at(0.05, centred=c), "draw it up", 1.0),
        fix(_flask_at(0.08, centred=c), "draw it up", 1.0),
        fix(_flask_at(0.10, centred=c), "draw it up", 0.8),
        hold(0.4, "let it settle"),
        Pose(0.6, _shift(0.0, 0.0, 0.0), FINGER_CLOSED, "square the tube upright", rot=_upright),
        hold(0.4, "let it settle"),
        fix(_flask_at(0.10, centred=c), "centre tube and bar", 0.5),
        fix(_flask_at(0.103, centred=c), "through the neck", 0.25, 0.6),
        fix(_flask_at(0.106, centred=c), "through the neck", 0.25, 0.6),
        fix(_flask_at(0.109, centred=c), "through the neck", 0.25, 0.6),
        fix(_flask_at(0.112, centred=c), "through the neck", 0.25, 0.6),
        fix(_flask_at(0.115, centred=c), "through the neck", 0.25, 0.6),
        fix(_flask_at(0.118, centred=c), "through the neck", 0.25, 0.6),
        fix(_flask_at(0.121, centred=c), "through the neck", 0.25, 0.6),
        fix(_flask_at(0.124, centred=c), "through the neck", 0.25, 0.6),
        fix(_flask_at(0.127, centred=c), "through the neck", 0.25, 0.6),
        fix(_flask_at(0.130, centred=c), "through the neck", 0.25, 0.6),
        fix(_flask_at(0.133, centred=c), "through the neck", 0.25, 0.6),
        fix(_flask_at(0.136, centred=c), "through the neck", 0.25, 0.6),
        fix(_flask_at(0.139, centred=c), "through the neck", 0.25, 0.6),
        fix(_flask_at(0.142, centred=c), "through the neck", 0.25, 0.6),
        fix(_flask_at(0.145, centred=c), "through the neck", 0.25, 0.6),
        fix(_flask_at(0.148, centred=c), "through the neck", 0.25, 0.6),
        fix(_flask_at(0.151, centred=c), "through the neck", 0.25, 0.6),
        fix(_flask_at(0.154, centred=c), "through the neck", 0.25, 0.6),
        fix(_flask_at(0.157, centred=c), "through the neck", 0.25, 0.6),
        fix(_flask_at(0.160, centred=c), "through the neck", 0.25, 0.6),
        fix(_flask_at(0.163, centred=c), "through the neck", 0.25, 0.6),
        fix(_flask_at(0.166, centred=c), "through the neck", 0.25, 0.6),
        fix(_flask_at(0.169, centred=c), "through the neck", 0.25, 0.6),
        fix(_flask_at(0.172, centred=c), "through the neck", 0.25, 0.6),
        fix(_flask_at(0.175, centred=c), "through the neck", 0.25, 0.6),
        fix(_flask_at(0.178, centred=c), "through the neck", 0.25, 0.6),
        fix(_flask_at(0.181, centred=c), "through the neck", 0.25, 0.6),
        fix(_flask_at(0.184, centred=c), "clear of the rim", 0.25, 0.6),
        fix(_flask_at(0.187, centred=c), "clear of the rim", 0.25, 0.6),
        fix(_flask_at(0.190, centred=c), "clear of the rim", 0.25, 0.6),
        fix(_flask_at(0.193, centred=c), "clear of the rim", 0.25, 0.6),
        fix(_flask_at(0.196, centred=c), "clear of the rim", 0.25, 0.6),
        fix(_flask_at(0.199, centred=c), "clear of the rim", 0.25, 0.6),
        fix(_flask_at(0.202, centred=c), "clear of the rim", 0.25, 0.6),
        fix(_flask_at(0.205, centred=c), "clear of the rim", 0.25, 0.6),
        fix(_flask_at(0.208, centred=c), "clear of the rim", 0.25, 0.6),
        fix(_beaker_at(RIM + 0.03), "over the beaker", 2.5),
        hold(0.6, "let it settle"),
        fix(_beaker_at(sa.BEAKER_H + 0.02), "down to the beaker", 1.5),
        fix(_beaker_at(0.045), "into the beaker", 1.5),
        fix(_beaker_at(0.045), "into the beaker", 0.8),
        hold(sum(w.duration for w in _left_after_wait()) + 0.3, "hold the tube still"),
        fix(_beaker_at(sa.BEAKER_H + 0.015), "tube up out of the beaker", 1.2),
        Pose(0.8, _shift(-0.06, 0.06, 0.0), FINGER_CLOSED, "aside, clear of the beaker", rot=_keep),
        hold(2.0, "hold"),
    ]
    return w


def _left_after_wait() -> list:
    """The left hand's part, from the moment the right hand holds the tube still in the beaker until it is back in its
    ready pose (the right hand keeps the tube there that long, then lifts it out and moves it aside)."""
    return [
        Pose(1.5, _ring(0.0, 0.06), OPEN_RING, "beside the ring", rot=RING_GRIP),
        Pose(0.5, _shift(0.0, 0.0, 0.0), OPEN_RING, "let it settle", rot=RING_GRIP),
        Pose(0.6, _ring_fix(0.0, 0.06), OPEN_RING, "line up beside the ring", rot=RING_GRIP),
        Pose(1.0, _ring_fix(0.0), OPEN_RING, "jaws round the ring", rot=RING_GRIP),
        Pose(0.6, _ring_fix(0.0), OPEN_RING, "jaws round the ring", rot=RING_GRIP),
        Pose(0.6, _shift(0.0, 0.0, 0.0), FINGER_CLOSED, "grip the ring", rot=RING_GRIP),
        Pose(1.2, _shift(0.0, 0.0, sa.ROD_TRAVEL + 0.004), FINGER_CLOSED, "slide the magnet up", rot=RING_GRIP),
        Pose(0.6, _shift(0.0, 0.0, 0.0), OPEN_RING, "let go of the ring", rot=RING_GRIP),
        Pose(1.5, _shift(0.0, 0.07, 0.0), OPEN_RING, "hand away", rot=RING_GRIP),
        Joints(2.5, (0.0, 0.0, 0.0), FINGER_CLOSED, "back to ready", q=tuple(READY)),
    ]


def _left() -> list:
    right = _right()
    t_wait = sum(w.duration for w in right[:[w.label for w in right].index("hold the tube still")])
    return [Joints(t_wait, (0.0, 0.0, 0.0), FINGER_CLOSED, "wait", q=tuple(READY)), *_left_after_wait()]


def demo():
    """The right hand takes the retriever lying on the table, turns it upright, steers the tube down through the flask's
    neck to the bar (the magnet picks it up against the tube's side), centres tube and bar in the neck, draws them out
    and holds the tube in the beaker; the left hand takes the ring and slides the magnet up, so the bar drops into the
    beaker. TASK_DEMO=strip: the same, but the tube comes up on its axis, not centred with the bar, so the neck scrapes
    the bar off."""
    return {"left": _left(), "right": _right()}
