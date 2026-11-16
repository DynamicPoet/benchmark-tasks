"""A folded manual wheelchair stands on the floor in front of the robot, facing it.

It is a standard lightweight folding wheelchair with an 18 in (457 mm) seat: 24 in rear wheels with handrims, 8 in
swivel casters, flip-back padded armrests, push handles, a black fabric seat and back, and an X-shaped cross-brace
under the seat. Folded, its two side frames stand close together (28 cm wide over the handrims) with the seat rails
raised between them. Opened, the seat rails rest in saddles on the side frames (61 cm wide over the handrims). It stands free on its
four wheels on a vinyl floor, its front casters' swivels 25 cm in front of the arms' bases, centred between the arms.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # helpers beside this file

import chair  # noqa: E402
import scenekit  # noqa: E402
import steptime  # noqa: E402
from yam_arena.constants import FINGER_CLOSED  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.station import FLOOR_Z  # noqa: E402

SCENE = "station"  # the station's rig; run it with --embodiment yam_floor_bay (the room without its table)

CHAIR_X = 0.68  # world x of the rear-hub midpoint
USD = chair.write()
CHAIR = scenekit.UsdBody(
    "wheelchair", pos=(CHAIR_X, 0.0, FLOOR_Z + 0.001), size=(1.0, 0.63, 0.93), mass=round(chair.TOTAL_MASS, 2),
    usd=str(USD.relative_to(chair.HERE)), articulation=True, joint_pos=((".*", 0.0),), friction=None,
    solver_iters=32,
    actuators=((chair.FABRIC_JOINTS, 20.0, 0.5),
               (("pivot_.*", "centre_pin"), 0.0, 2.0),  # the X's bushings, as damping
               (("axle_.*", "swivel_.*", "caster_axle_.*"), 0.0, 0.02)),
)
OBJECTS = [CHAIR]

# -- scoring constants (the spec's numbers) -----------------------------------------------------------------------
MOVE_FAIL, TURN_FAIL = 0.20, 20.0  # m, deg: zero at any moment, and it stays zero
MOVE_OK, TURN_OK = 0.05, 5.0  # m, deg: in place at the end
SEAT_GAP = 0.003  # m: a seat rail rests in a saddle when its gap is at most this
SEAT_SIDE = 0.012  # m: and its centre is within this of the saddle's centre line
TOUCH = 0.01  # m: a part this close to the floor touches it
CLEAR = 0.05  # m: fingertips this far from the chair
STILL = 0.005  # m over the last 1 s
CONTROL_HZ = steptime.CONTROL_HZ
WINDOW = int(round(CONTROL_HZ)) + 1  # samples spanning 1 s
GAP_FOLDED, GAP_OPEN = 2 * chair.W_FOLD, 2 * chair.W_OPEN

_BODIES = {b.name: b for b in chair.bodies()}
_SOLIDS = {name: b.colliders + b.probes for name, b in _BODIES.items() if b.colliders or b.probes}  # incl. fabric


# -- geometry ---------------------------------------------------------------------------------------------------------


def _rot(q_xyzw) -> np.ndarray:
    x, y, z, w = q_xyzw
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _to_world(pose, pts) -> np.ndarray:
    """Points in a chair body's own (authored) frame to world, given the body's pose [x y z qx qy qz qw]."""
    return np.atleast_2d(np.asarray(pts, float)) @ _rot(pose[3:]).T + pose[:3]


def _to_local(pose, pts) -> np.ndarray:
    return (np.atleast_2d(pts) - pose[:3]) @ _rot(pose[3:])


def _dist_segment(p, a, b) -> np.ndarray:
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def _dist_to_solids(name: str, pose, pts_world) -> float:
    """Smallest distance from world points to a chair body's collision solids (capsules, boxes, wheels)."""
    p = _to_local(pose, pts_world)
    best = np.inf
    for c in _SOLIDS[name]:
        if c.kind == "capsule":
            r, h = c.size
            ax = chair.quat_to_matrix_wxyz(c.quat)[:, 2]
            d = _dist_segment(p, c.center - ax * h / 2, c.center + ax * h / 2) - r
        elif c.kind == "box":
            rot = chair.quat_to_matrix_wxyz(c.quat)
            q = (p - c.center) @ rot
            d = np.linalg.norm(np.maximum(np.abs(q) - np.asarray(c.size) / 2, 0.0), axis=1)
        else:  # a wheel: cylinder about y
            pts = np.asarray(c.size)
            centre = pts.mean(axis=0)
            radius = np.max(np.linalg.norm((pts - centre)[:, [0, 2]], axis=1))
            half = (pts[:, 1].max() - pts[:, 1].min()) / 2
            q = p - centre
            radial = np.maximum(np.hypot(q[:, 0], q[:, 2]) - radius, 0.0)
            axial = np.maximum(np.abs(q[:, 1]) - half, 0.0)
            d = np.hypot(radial, axial)
        best = min(best, float(d.min()))
    return best


def _fingertips(env) -> np.ndarray:
    """Points along both fingers of both grippers, from the pad tips up 4 cm (world)."""
    out = []
    for side in ("left", "right"):
        links = scenekit.body_poses(env, f"{side}_robot")
        g = links["gripper"]
        rg = _rot(g[3:])
        for tip in ("tip_left", "tip_right"):
            local = rg.T @ (links[tip][:3] - g[:3])
            inward = -np.sign(local[0]) if abs(local[0]) > 1e-6 else 0.0
            for dx in (0.0, 0.015, 0.03, 0.044):
                for z in (0.147, 0.127, 0.107):
                    out.append(g[:3] + rg @ np.array([local[0] + inward * dx, local[1], z]))
    return np.array(out)


# -- the trial-long tracker ---------------------------------------------------------------------------------------------

_T: dict = {}


def _clear() -> None:
    _T.clear()
    _T.update(start=None, failed=False, why="", history=[], full_steps=0, step=0, m=None)


_clear()


@scenekit.on_reset
def _reset(env) -> None:
    """A reset puts the chair back folded in its place in one step; only a reset clears a failure."""
    _clear()
    _fabric(env, reset=True)


def _fabric(env, reset: bool = False) -> None:
    """The fabric halves follow the frames: each pair meets in the middle, slack or flat (visual only, no load)."""
    asset = env.unwrapped.scene["wheelchair"]
    ids, names = asset.find_joints(list(chair.FABRIC_JOINTS), preserve_order=True)
    if reset:
        target = [0.0] * len(ids)
    else:
        P = scenekit.body_poses(env, "wheelchair")
        rail = {s: _to_world(P[f"brace_{s}"], np.mean(chair.RAIL_ENDS[s], axis=0))[0] for s in chair.SIDES}
        cane = {s: _to_world(P[f"frame_{s}"], [chair.X_CANE, chair.SIDES[s] * chair.W_FOLD, 0.7])[0] for s in chair.SIDES}
        seat_b = chair.seat_fold_angle(float(np.linalg.norm((rail["l"] - rail["r"])[:2])))
        back_b = chair.back_fold_angle(float(np.linalg.norm((cane["l"] - cane["r"])[:2])))
        target = []
        for n in names:
            kind, _, side = n.split("_")
            s = chair.SIDES[side]
            if kind == "seat":
                r = _rot(P[f"brace_{side}"][3:])
                roll = math.atan2(r[2, 1], r[1, 1])
                target.append(s * (seat_b - chair.FOLD_SEAT) - roll)
            else:
                target.append(s * (back_b - chair.FOLD_BACK))
    import torch

    t = torch.tensor([target], dtype=torch.float32, device=asset.device)
    asset.set_joint_position_target(t, joint_ids=ids)


@scenekit.on_step
def _measure(env) -> None:
    """Everything the score needs, after every control step (before success() and score() read it)."""
    P = scenekit.body_poses(env, "wheelchair")
    m: dict = {}
    hub = {s: _to_world(P[f"frame_{s}"], chair.HUB[s])[0] for s in chair.SIDES}
    mid = (hub["l"] + hub["r"]) / 2
    line = hub["l"] - hub["r"]
    heading = math.degrees(math.atan2(line[1], line[0]))
    if _T["start"] is None:
        _T["start"] = (mid.copy(), heading)
    start, start_heading = _T["start"]
    m["moved"] = float(np.linalg.norm((mid - start)[:2]))
    m["turned"] = float(abs((heading - start_heading + 180.0) % 360.0 - 180.0))
    rail = {s: _to_world(P[f"brace_{s}"], np.mean(chair.RAIL_ENDS[s], axis=0))[0] for s in chair.SIDES}
    m["rail_gap"] = float(np.linalg.norm((rail["l"] - rail["r"])[:2]))
    m["opened"] = (m["rail_gap"] - GAP_FOLDED) / (GAP_OPEN - GAP_FOLDED)
    # seat rails in their saddles, measured in each side frame
    gaps, sides_off = [], []
    for s, sign in chair.SIDES.items():
        ends = _to_local(P[f"frame_{s}"], _to_world(P[f"brace_{s}"], np.array(chair.RAIL_ENDS[s])))
        for sx in chair.SADDLES_X:
            t = (sx - ends[0, 0]) / (ends[1, 0] - ends[0, 0])
            c = ends[0] + t * (ends[1] - ends[0])
            gaps.append(float(c[2] - chair.R_RAIL - chair.SADDLE_TOP))
            sides_off.append(float(c[1] - sign * chair.W_FOLD))
    m["saddle_gaps"], m["saddle_side"] = gaps, sides_off
    m["seated"] = all(-0.005 <= g <= SEAT_GAP for g in gaps) and all(abs(o) <= SEAT_SIDE for o in sides_off)
    # wheels on the floor; any other part on the floor means the chair has tipped over
    low = {}
    for s in chair.SIDES:
        a = _rot(P[f"frame_{s}"][3:]) @ np.array([0.0, 1.0, 0.0])
        low[f"wheel_{s}"] = float(hub[s][2] - chair.R_WHEEL * math.sqrt(max(0.0, 1 - a[2] ** 2)) - FLOOR_Z)
        cp = P[f"caster_{s}"]
        cc = _to_world(cp, [chair.X_FRONT + chair.CASTER_TRAIL, chair.SIDES[s] * chair.W_FOLD, chair.R_CASTER])[0]
        ca = _rot(cp[3:]) @ np.array([0.0, 1.0, 0.0])
        low[f"caster_{s}"] = float(cc[2] - chair.R_CASTER * math.sqrt(max(0.0, 1 - ca[2] ** 2)) - FLOOR_Z)
    m["wheel_low"] = low
    m["on_wheels"] = all(v <= TOUCH for v in low.values())
    parts = [_to_world(P[f"frame_{s}"], chair.frame_points(s)) for s in chair.SIDES]
    parts += [_to_world(P[f"brace_{s}"], chair.brace_points(s)) for s in chair.SIDES]
    m["part_low"] = float(np.min(np.vstack(parts)[:, 2]) - FLOOR_Z)
    m["tipped"] = m["part_low"] <= TOUCH
    # fingertips
    tips = _fingertips(env)
    m["finger_gap"] = min(_dist_to_solids(n, P[n], tips) for n in _SOLIDS)
    # stillness: hub midpoint and both seat rails over the last 1 s
    hist = _T["history"]
    hist.append(np.concatenate([mid, rail["l"], rail["r"]]))
    del hist[:-WINDOW]
    if len(hist) == WINDOW:
        h = np.array(hist).reshape(WINDOW, 3, 3)
        spread = max(float(np.max(np.linalg.norm(h[:, None, k] - h[None, :, k], axis=-1))) for k in range(3))
    else:
        spread = np.inf
    m["still_spread"] = spread
    m["still"] = spread < STILL
    # failures that stay failed
    if not _T["failed"]:
        for why, bad in (("tipped over", m["tipped"]), ("moved over 20 cm", m["moved"] > MOVE_FAIL),
                         ("turned over 20 degrees", m["turned"] > TURN_FAIL)):
            if bad:
                _T["failed"], _T["why"] = True, why
                break
    m["placed"] = m["moved"] <= MOVE_OK and m["turned"] <= TURN_OK
    m["full"] = (not _T["failed"] and m["opened"] >= 0.5 and m["seated"] and m["placed"] and m["on_wheels"]
                 and m["finger_gap"] >= CLEAR and m["still"])
    _T["full_steps"] = _T["full_steps"] + 1 if m["full"] else 0
    _T["step"] += 1
    _T["m"] = m
    _fabric(env)
    _log(m)


def _log(m) -> None:
    out = os.environ.get("TASK_SCORE_DIR")
    if not out:
        return
    os.makedirs(out, exist_ok=True)
    row = {"step": _T["step"], "t": round(_T["step"] / CONTROL_HZ, 3), "failed": _T["failed"], "why": _T["why"],
           "score": _score_now()}
    for k, v in m.items():
        if isinstance(v, (bool, np.bool_)):
            row[k] = bool(v)
        elif isinstance(v, float):
            row[k] = round(v, 5) if math.isfinite(v) else None
        elif isinstance(v, list):
            row[k] = [round(x, 5) for x in v]
        elif isinstance(v, dict):
            row[k] = {a: round(b, 5) for a, b in v.items()}
    env = scenekit.env()
    if env is not None:
        P = scenekit.body_poses(env, "wheelchair")
        for name in ("frame_l", "frame_r", "brace_l", "brace_r"):
            r = _rot(P[name][3:])
            row[f"roll_{name}"] = round(math.degrees(math.atan2(r[2, 1], r[1, 1])), 2)
        for s in chair.SIDES:
            piv = _to_world(P[f"frame_{s}"], [chair.X_MID, chair.SIDES[s] * chair.W_FOLD, chair.Z_PIVOT])[0]
            row[f"pivot_{s}"] = [round(float(x), 4) for x in piv]
        row["hub_mid"] = [round(float(x), 4) for x in _T["history"][-1][:3]]
        asset = env.unwrapped.scene["wheelchair"]
        q = asset.data.joint_pos.torch[0].cpu().numpy()
        qt = asset.data.joint_pos_target.torch[0].cpu().numpy()
        row["joints"] = {n: round(math.degrees(float(q[i])), 2) for i, n in enumerate(asset.joint_names)
                         if not n.startswith(("axle", "caster_axle"))}
        row["fabric_target"] = {n: round(math.degrees(float(qt[i])), 2) for i, n in enumerate(asset.joint_names)
                                if "hinge" in n}
        for side in ("left", "right"):
            links = scenekit.body_poses(env, f"{side}_robot")
            g = links["gripper"]
            row[f"tcp_{side}"] = [round(float(x), 5) for x in g[:3] + _rot(g[3:]) @ np.array([0, 0, 0.134])]
    with open(os.path.join(out, "telemetry.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")


def _score_now() -> float:
    m = _T["m"]
    if m is None or _T["failed"] or m["opened"] < 0.5:
        return 0.0
    if not m["seated"]:
        return 0.25
    if not m["placed"]:
        return 0.5
    if m["full"] and _T["full_steps"] >= WINDOW - 1:
        return 1.0
    return 0.75


def score(objects) -> float:
    """0 at any moment the chair tips over, moves over 20 cm or turns over 20 degrees, and it stays 0. Otherwise, at
    the end: 0 if the seat rails opened less than half way; 0.25 if opened half way but not both seat rails resting in
    their saddles (gap at most 3 mm); 0.5 if seated but the chair moved over 5 cm or turned over 5 degrees; 0.75 if
    seated and in place but a fingertip is within 5 cm of the chair, a wheel is off the floor or the chair moved 5 mm
    or more in the last second; 1.0 when all of that holds for 1 s."""
    return _score_now()


def success(objects) -> bool:
    steptime.tick()  # the harness calls success() once per control step
    m = _T["m"]
    return bool(m is not None and m["full"])


# -- the scripted demo ------------------------------------------------------------------------------------------------

PRESS_X = CHAIR_X - 0.30  # grip each seat rail 30 cm behind its front end, ahead of the armrest
GRIP_OPEN = -0.022  # fingers part opened (44 mm), narrow enough to pass between the folded rails
SPREAD_S = 10.0


def _rail_z(w: float) -> float:
    """World height of a seat rail's centre when the side frames are w from the middle (the X's geometry)."""
    return FLOOR_Z + chair.Z_PIVOT + math.sqrt(chair.L_BRACE**2 - (2 * w) ** 2)


def _plan(sign: float, travel: float, settle: float = 0.003) -> list[Waypoint]:
    """One arm: grip its seat rail, then spread and press it along the X's own path (the policy keeps the gripper on
    that path from the arm's joint angles), hold it seated, ease off, let go, rise and back away."""
    w0, w1 = chair.W_FOLD, chair.W_FOLD + travel
    z0 = _rail_z(w0)
    path = []
    for k, frac in enumerate((0.35, 0.7, 1.0)):
        w = w0 + frac * (w1 - w0)
        path.append(Waypoint(SPREAD_S / 3, (PRESS_X, sign * w, _rail_z(min(w, chair.W_OPEN))), FINGER_CLOSED,
                             f"track: spread and press {k + 1}"))
    end = (PRESS_X, sign * w1, _rail_z(min(w1, chair.W_OPEN)))
    seat = (end[0], end[1], end[2] - settle)
    return [
        Waypoint(2.0, (PRESS_X, sign * w0, z0 + 0.10), GRIP_OPEN, "above the seat rail"),
        Waypoint(1.5, (PRESS_X, sign * w0, z0), GRIP_OPEN, "down onto the rail"),
        Waypoint(0.8, (PRESS_X, sign * w0, z0), FINGER_CLOSED, "grip the rail"),
        *path,
        Waypoint(1.5, seat, FINGER_CLOSED, "track: hold it seated"),
        Waypoint(2.0, end, FINGER_CLOSED, "ease off"),
        Waypoint(0.6, end, GRIP_OPEN, "let go"),
        Waypoint(1.5, (end[0], end[1], end[2] + 0.25), GRIP_OPEN, "rise"),
        Waypoint(2.0, (0.28, sign * 0.30, 0.90), GRIP_OPEN, "back away"),
    ]


def _hold(sign: float) -> list[Waypoint]:
    """Grip the rail and only hold it where it is, without feedback (the careless partner)."""
    w0 = chair.W_FOLD
    z0 = _rail_z(w0)
    grip = (PRESS_X, sign * w0, z0)
    return [
        Waypoint(2.0, (PRESS_X, sign * w0, z0 + 0.10), GRIP_OPEN, "above the seat rail"),
        Waypoint(1.5, grip, GRIP_OPEN, "down onto the rail"),
        Waypoint(0.8, grip, FINGER_CLOSED, "grip the rail"),
        Waypoint(SPREAD_S + 3.5, grip, FINGER_CLOSED, "hold"),
        Waypoint(0.6, grip, GRIP_OPEN, "let go"),
        Waypoint(1.5, (grip[0], grip[1], grip[2] + 0.15), GRIP_OPEN, "rise"),
        Waypoint(2.0, (0.28, sign * 0.30, 0.90), GRIP_OPEN, "back away"),
    ]


def demo():
    """Both arms grip the two seat rails, then spread and press them down together along mirror-image paths, at the
    same speed, until the rails sit in their saddles; they ease off, let go and back away. Open loop.
    TASK_DEMO=failure: the left arm pulls its side all the way while the right arm only holds its rail."""
    travel = chair.W_OPEN - chair.W_FOLD
    if os.environ.get("TASK_DEMO") == "failure":
        return {"left": _plan(1.0, 2 * travel), "right": _hold(-1.0)}
    return {"left": _plan(1.0, travel), "right": _plan(-1.0, travel)}
