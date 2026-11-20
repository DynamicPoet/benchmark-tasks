"""gatekit: the scoring checks ("gates") a task's scorer keeps needing, each reading the real event and not something a lazy policy can do instead.

Pure Python and NumPy, no simulator: copy it beside task.py with demokit.py and mechkit.py, and a scorer and its check_score.py both import it.
Every gate reads the same state a task file already reads:

    import gatekit as gk
    import mechkit as m

    tips = m.fingertips(env)              # (24, 3): the left hand's 12 points, then the right's; every gate that takes them wants all 24
    part = m.state(env, "gun")            # pos, rotation(), to_local(), to_world(), lin_vel, ang_vel, joint()

    GRIP = gk.Grip(half=0.014, across="x", lo=(-1, -0.13, -0.115), hi=(1, -0.07, -0.025))    # where the part is held, in its own frame
    held = gk.held(tips, part, GRIP, resting=gk.on_support(underside, STANDS))                  # a grasp, never nearness
    clear = gk.hands_clear(tips, gk.along(part, (0, 0, 0), (0, -0.18, 0)), 0.05)               # every fingertip of both hands, the whole part

The stateful gates (`HoldThroughout`, `Latch`, `Peak`, `NetRotation`, `PathInsert`, `Strike`, `EndHold`) each have `reset(env=None)`, so that one
instance can be registered with `m.on_reset(gate.reset)` and stepped from the task's `on_step` hook. `Ladder` and `check_ladder` build and check the
partial-credit table. `Pose`, `hand` and `tips` make hand-built states for check_score.py. The `lazy_*` generators are cheat modes for a demo script.

Lengths in metres, angles in radians unless a name says degrees, all points in the station frame that mechkit reports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

# -- fingertip points ------------------------------------------------------------------------------------------------------------------------------
# mechkit.fingertips(env) gives 24 points: the left hand's 12, then the right's. In each hand the first 6 lie on one finger and the next 6 on the other,
# at the fingertip and 3 cm up the finger, at the middle and both edges of its width, in the same order on both fingers. Each point lies on the finger's
# centre line, PAD_INSET outside its pad's face, so a hand's jaw gap (pad to pad) is the distance between matching points less 2 x PAD_INSET.

PAD_INSET = 0.0045
FINGER_DY = (-0.018, 0.0, 0.018)  # across the finger's width
FINGER_DZ = (0.147, 0.117)  # along the gripper's axis, from its body's origin: the tip, and 3 cm up the finger
HANDS = {"left": slice(0, 12), "right": slice(12, 24)}
_AXIS = {"x": 0, "y": 1, "z": 2}


def all_tips(tips) -> np.ndarray:
    """The 24 fingertip points as an array. Anything else raises: a gate fed one hand's points misses what the other hand does."""
    t = np.asarray(tips, dtype=float)
    if t.shape != (24, 3):
        raise ValueError(f"pass all 24 fingertip points of both hands (mechkit.fingertips(env)), not an array of shape {t.shape}")
    return t


def fingers(tips, side: str) -> tuple[np.ndarray, np.ndarray]:
    """One hand's two fingers, 6 points each."""
    h = all_tips(tips)[HANDS[side]]
    return h[:6], h[6:]


def jaw_gap(tips, side: str) -> float:
    """How far apart one hand's pads are now, read from its fingertip points: what its jaws are open to."""
    a, b = fingers(tips, side)
    return float(np.mean(np.linalg.norm(a - b, axis=1))) - 2 * PAD_INSET


def hand_mask(*sides: str) -> np.ndarray:
    """A mask over the 24 points, True on the named hands' points: for `Strike.step(skip=...)`."""
    mask = np.zeros(24, dtype=bool)
    for s in sides:
        mask[HANDS[s]] = True
    return mask


# -- held: a grasp, never nearness ------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Grip:
    """Where a part can be held, in the part's own frame.

    half: half the part's width across the jaws there; for a round part, its radius.
    across: a flat grip: the part's axis ("x", "y" or "z") the jaws close along. Leave it None for a round grip, which the jaws may close across
        from any side.
    axis: a round grip's own axis ("x", "y" or "z").
    centre: a point on the grip's middle plane (flat) or on its axis (round), in the part's frame.
    lo, hi: a box in the part's frame that a finger's points must be inside to count: the grip's length and width. Leave the coordinate across the
        jaws (a flat grip's `across`, a round grip's two radial ones) open.
    air: how far a pad may stand off the part's surface and still count as on it.
    press: how far a pad may sink into the part's collider (contact penetration). Never more than half of `half`, so jaws shut on nothing fail.
    """

    half: float
    across: str | None = None
    axis: str = "z"
    centre: tuple = (0.0, 0.0, 0.0)
    lo: tuple = (-math.inf, -math.inf, -math.inf)
    hi: tuple = (math.inf, math.inf, math.inf)
    air: float = 0.003
    press: float = 0.008

    def window(self) -> tuple[float, float, float, float]:
        """(nearest, farthest) a finger point may be from the grip's middle plane or axis, and (narrowest, widest) the jaw gap may be."""
        press = min(self.press, 0.5 * self.half)
        return (self.half - press + PAD_INSET, self.half + self.air + PAD_INSET, 2 * (self.half - press), 2 * (self.half + self.air))


def grasps(local12, gap: float, grip: Grip) -> bool:
    """Whether one hand grasps the grip: its 12 points (in the part's frame) put a finger on each side of the part, each pad within `grip.air` of the
    surface (or pressed into it), and its jaw gap `gap` fits the part's width. Most callers want `held`."""
    p = np.asarray(local12, dtype=float)
    near, far, narrow, wide = grip.window()
    if not narrow <= gap <= wide:
        return False  # jaws open round the part, or shut past it
    lo, hi, c = np.asarray(grip.lo, float), np.asarray(grip.hi, float), np.asarray(grip.centre, float)
    if grip.across is not None:
        k = _AXIS[grip.across]
        other = [i for i in range(3) if i != k]
        inbox = np.all((p[:, other] >= lo[other]) & (p[:, other] <= hi[other]), axis=1)
        s = p[:, k] - c[k]

        def side(rows: slice, sign: float) -> bool:
            """The finger is on that side (its points in the region lie more than half of `half` out there on average, so the edges of one
            finger's width never count for both sides), with a point on the surface there."""
            inb = inbox[rows]
            return bool(inb.any() and sign * float(s[rows][inb].mean()) > 0.5 * grip.half
                        and np.any(inb & (sign * s[rows] > near) & (sign * s[rows] < far)))

        a, b = slice(0, 6), slice(6, 12)
        return (side(a, 1.0) and side(b, -1.0)) or (side(a, -1.0) and side(b, 1.0))
    k = _AXIS[grip.axis]
    inbox = np.all((p >= lo) & (p <= hi), axis=1)
    rv = p - c
    rv[:, k] = 0.0
    r = np.linalg.norm(rv, axis=1)
    ok = inbox & (r > near) & (r < far)
    if not ok[:6].any() or not ok[6:].any():
        return False
    da = (rv[:6][ok[:6]] / r[:6][ok[:6], None]).mean(axis=0)
    db = (rv[6:][ok[6:]] / r[6:][ok[6:], None]).mean(axis=0)
    return float(da @ db) <= -0.5 * float(np.linalg.norm(da) * np.linalg.norm(db))  # on opposite sides: 120 degrees or more apart round the axis


def holding(tips, part, grip: Grip, *, resting: bool, hands: Sequence[str] = ("left", "right")) -> tuple[str, ...]:
    """The hands that hold `part` now (see `held`), of `hands`."""
    if resting:
        return ()
    t = all_tips(tips)
    loc = part.to_local(t)
    return tuple(h for h in hands if grasps(loc[HANDS[h]], jaw_gap(t, h), grip))


def held(tips, part, grip: Grip, *, resting: bool, hand: str | None = None) -> bool:
    """A real grasp: the fingertip points of both fingers of one hand on opposite sides of the part, within `grip`'s region; each pad within
    `grip.air` (3 mm) of the part's surface, or pressed into it; the jaws closed to the part's width (the gap read from the points); and the part off
    its rest (`resting` False). `hand` restricts it to "left" or "right"; by default either hand counts, and both are always checked.

    Reads: the jaws shut on the part, as a camera on the hand would show.
    Blocks: an open gripper hovering round the part; one fingertip near or on it; jaws hovering a few millimetres off it; jaws shut beside it or
    through a part with no collider; a part nudged along or still lying on its stand; the hand the text does not name (with `hand`).

    `resting` is required so that it is always decided: pass `on_support(...)`, a height test, or False for a part that never rests (a fixed one
    squeezed in place), with the reason in a comment.
    """
    return bool(holding(tips, part, grip, resting=resting, hands=(hand,) if hand else ("left", "right")))


def on_support(points, supports: Iterable[Sequence[float]], gap: float = 0.01) -> bool:
    """Whether a part rests on a support: any of `points` (world points on its underside) over a support and less than `gap` above its top.
    supports: (cx, cy, hx, hy, top) boxes, their centres, half sizes (with any margin) and top heights. A part slid along its cradles or lifted a few
    millimetres is still on them; only one clear of every support is off its rest."""
    p = np.atleast_2d(np.asarray(points, dtype=float))
    for cx, cy, hx, hy, top in supports:
        over = (np.abs(p[:, 0] - cx) < hx) & (np.abs(p[:, 1] - cy) < hy)
        if np.any(over & (p[:, 2] - top < gap)):
            return True
    return False


class HoldThroughout:
    """A hold the text says lasts ("held from the first turn until it is out"), checked on every step of that interval.

    Each step: `step(holding, active)`, with `active` True on the steps the hold must last (from its start to its end, both read from real events)
    and `holding` from `held(...)`. The first active step without the hold latches `broken`, which stays.
    Reads: the hold at every step of the interval. Blocks: a hold sampled once (a grasp at the start, or as the event passes some angle), with the hand
    let go between samples. `ok` is False until the interval has begun: a hold never checked is not a hold.
    """

    def __init__(self):
        self.reset()

    def reset(self, env=None) -> None:
        self.steps, self.broken, self.broke_at, self._n = 0, False, None, 0

    def step(self, holding: bool, active: bool) -> bool:
        self._n += 1
        if active:
            self.steps += 1
            if not holding and not self.broken:
                self.broken, self.broke_at = True, self._n
        return self.ok

    @property
    def ok(self) -> bool:
        return self.steps > 0 and not self.broken


# -- hands clear ------------------------------------------------------------------------------------------------------------------------------------


def along(part, a, b, n: int = 7) -> np.ndarray:
    """`n` world points on the segment from `a` to `b` in `part`'s own frame: a part's centre line, for `hands_clear`."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return part.to_world(np.array([a + (b - a) * u for u in np.linspace(0.0, 1.0, n)]))


def _to_polyline(x: np.ndarray, line: np.ndarray, joined: bool) -> np.ndarray:
    """Each point of x's distance to the polyline through `line` (or to its points alone)."""
    d = np.linalg.norm(x[:, None, :] - line[None, :, :], axis=2).min(axis=1)
    if not joined or len(line) < 2:
        return d
    p, q = line[:-1], line[1:]
    seg = q - p
    u = np.clip(np.einsum("nsk,sk->ns", x[:, None, :] - p[None], seg) / np.maximum((seg * seg).sum(axis=1), 1e-18), 0.0, 1.0)
    foot = p[None] + u[..., None] * seg[None]
    return np.minimum(d, np.linalg.norm(x[:, None, :] - foot, axis=2).min(axis=1))


def _lines(lines) -> list[np.ndarray]:
    """One line of points, or a list of lines."""
    if isinstance(lines, np.ndarray):
        return [np.atleast_2d(lines.astype(float))]
    lines = list(lines)
    if np.asarray(lines[0], dtype=float).ndim == 1:
        return [np.atleast_2d(np.asarray(lines, dtype=float))]
    return [np.atleast_2d(np.asarray(L, dtype=float)) for L in lines]


def min_distance(tips, lines, radius: float = 0.0, joined: bool = True) -> float:
    """The least distance from any of the 24 fingertip points to the object, less `radius` (the object's radius about its centre lines).
    lines: one (n, 3) array of world points along the object, or a list of them (one per limb: a body and a grip, a shaft and a head)."""
    t = all_tips(tips)
    return float(min(_to_polyline(t, L, joined).min() for L in _lines(lines))) - radius


def hands_clear(tips, lines, dist: float, radius: float = 0.0, joined: bool = True) -> bool:
    """Every fingertip point of both hands at least `dist` from the object, measured to points along the whole object.

    lines: world points along the object's centre lines (`along(...)`), one array or a list of them; with `joined` (the default) the distance is to the
    segments between neighbouring points, so a fingertip cannot sit on the object between two samples; `radius` measures from the surface.
    Reads: no finger near any part of the object. Blocks: a part held by its far end (its top in a bin) while the gate looks at its middle; one hand
    clear and the other still pinning it; a check of the hand that did the work only.
    """
    return min_distance(tips, lines, radius, joined) > dist


# -- latches, peaks, holds --------------------------------------------------------------------------------------------------------------------------


class Latch:
    """A failure that stays failed: once `trip(True)`, it is on until a reset. Read it as a bool.
    Blocks: a run that drops, pierces, dents or splits and then recovers to a passing end state."""

    def __init__(self, why: str = ""):
        self.why = why
        self.reset()

    def reset(self, env=None) -> None:
        self.on, self.at = False, None

    def trip(self, cond, at=None) -> bool:
        if cond and not self.on:
            self.on, self.at = True, at
        return self.on

    def __bool__(self) -> bool:
        return self.on


class Peak:
    """The peak of a value (a speed, a force, a dose rate) over an event: `update(value, active)` every step, counting only active steps.
    Reads: the whole event. Blocks: a fault read from one step's sample (the exit step, the first contact), which a spike on the step before misses."""

    def __init__(self, floor: float = 0.0):
        self.floor = floor
        self.reset()

    def reset(self, env=None) -> None:
        self.value, self.samples = self.floor, 0

    def update(self, value, active: bool = True) -> float:
        if active:
            self.samples += 1
            self.value = max(self.value, float(value))
        return self.value


def rel_speed(a, b) -> float:
    """Body a's speed relative to body b, from their states' linear velocities: a cork against its bottle, a part against the hand's tool."""
    return float(np.linalg.norm(np.asarray(a.lin_vel, float) - np.asarray(b.lin_vel, float)))


class EndHold:
    """The scorer's own hold of the end state (with `SUCCESS_HOLD_S = 0`): `step(ok)` each step, `done` latches once ok has held `steps` running steps."""

    def __init__(self, steps: int):
        self.steps = int(steps)
        self.reset()

    def reset(self, env=None) -> None:
        self.count, self.done = 0, False

    def step(self, ok: bool) -> bool:
        self.count = self.count + 1 if ok else 0
        self.done = self.done or self.count >= self.steps
        return self.done


# -- net rotation -----------------------------------------------------------------------------------------------------------------------------------


def wrap(a: float) -> float:
    """An angle in radians into [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def quat_wxyz(R) -> np.ndarray:
    """A rotation matrix's unit quaternion (w, x, y, z)."""
    R = np.asarray(R, float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = 2.0 * math.sqrt(tr + 1.0)
        q = (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = ((R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s)
    q = np.array(q)
    return q / np.linalg.norm(q)


def twist(R, axis) -> float:
    """The twist of rotation R about the unit `axis` (in R's own frame), by swing-twist: the turn a mark on the body reads about that axis, whatever
    the axis has swung. Radians, in [-pi, pi)."""
    q = quat_wxyz(R)
    return wrap(2.0 * math.atan2(float(q[1:] @ np.asarray(axis, float)), float(q[0])))


def rot(axis, deg: float) -> np.ndarray:
    """The rotation by `deg` degrees about `axis` (a vector, or "x", "y", "z")."""
    a = np.eye(3)[_AXIS[axis]] if isinstance(axis, str) else np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    t = math.radians(deg)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(t) * K + (1.0 - math.cos(t)) * K @ K


class NetRotation:
    """The net signed rotation of a body about its own axis, unwrapped across steps: turns made back undo turns made forward.

    `step(R, ref=None, count=True)` every step, with R the body's rotation (`m.state(env, name).rotation()`) and `ref` the rotation of the body it turns
    against (the bottle a cork turns in, a knob's housing), or None for the world. Each step reads the body's twist about `axis` (in its own frame)
    against its pose at the first step, by swing-twist, so the reading depends on where the body is, not on the path it took; the change from the last
    step, wrapped to under half a turn, is added to `net` on the steps with `count` True. The reading is kept up to date on every step, so turns made
    while `count` is False are never counted later. If the axis swings more than 90 degrees from the reference, the reference moves to the body's
    pose there, so the reading never meets the half-turn swing where a twist is undefined.
    `add(angle, count=True)` instead takes an angle reading of the task's own (a joint, a mark's angle), in radians, and unwraps it the same way.

    Reads: where the body has turned to, net. Blocks: rocking back and forth counted as turns (summed absolute spin); a reading that jumps a full turn
    at +-180 degrees; a turn counted for the wrong body: with `ref`, a cork and bottle turned together read 0, and so does a tilted body swept round
    the vertical. Use `turns` or `deg` with their sign: a loop turned the wrong way reads negative.
    """

    def __init__(self, axis=(0.0, 0.0, 1.0)):
        a = np.eye(3)[_AXIS[axis]] if isinstance(axis, str) else np.asarray(axis, float)
        self.axis = a / np.linalg.norm(a)
        self.reset()

    def reset(self, env=None) -> None:
        self.net, self.base, self.last, self.last_angle = 0.0, None, 0.0, None

    def step(self, R, ref=None, count: bool = True) -> float:
        R = np.asarray(R, float)
        rel = R if ref is None else np.asarray(ref, float).T @ R
        if self.base is None:
            self.base, self.last = rel.copy(), 0.0
            return self.net
        d = self.base.T @ rel
        t = twist(d, self.axis)
        if count:
            self.net += wrap(t - self.last)
        self.last = t
        if float(self.axis @ d @ self.axis) < 0.0:  # swung more than 90 degrees: take the reference from here
            self.base, self.last = rel.copy(), 0.0
        return self.net

    def add(self, angle: float, count: bool = True) -> float:
        if self.last_angle is not None and count:
            self.net += wrap(angle - self.last_angle)
        self.last_angle = float(angle)
        return self.net

    @property
    def deg(self) -> float:
        return math.degrees(self.net)

    @property
    def turns(self) -> float:
        return self.net / (2 * math.pi)


# -- through, inserted, seated: a state reached by a tracked path ----------------------------------------------------------------------------------


class Face:
    """A face a tip must pass through, as it is now, in the world: a point on it (`origin`), a rotation whose columns are its x and y (its outline's
    axes) and its normal (out, toward where the tool comes from), and its outline: a rectangle from `lo` to `hi` in its x and y, or a circle of
    `radius` round its origin. `Face.on(part, ...)` places one fixed in a part's frame."""

    def __init__(self, origin, rot, lo=None, hi=None, radius: float | None = None):
        self.origin, self.rot = np.asarray(origin, float), np.asarray(rot, float)
        self.lo, self.hi, self.radius = lo, hi, radius
        if (lo is None or hi is None) and radius is None:
            raise ValueError("a face needs an outline: lo and hi, or radius")

    @classmethod
    def on(cls, part, origin, rot=np.eye(3), **outline) -> "Face":
        """A face fixed in `part`'s frame: `origin` and `rot` in the part's frame."""
        return cls(part.to_world(np.asarray(origin, float)[None])[0], part.rotation() @ np.asarray(rot, float), **outline)

    def local(self, p) -> np.ndarray:
        return self.rot.T @ (np.asarray(p, float) - self.origin)

    def inside(self, q) -> bool:
        if self.radius is not None:
            return bool(math.hypot(q[0], q[1]) < self.radius)
        return bool(self.lo[0] < q[0] < self.hi[0] and self.lo[1] < q[1] < self.hi[1])


class PathInsert:
    """"Through", "inserted" and "seated" as a state reached by a tracked path, for a tool or part whose tip may have no collider.

    Each step: `step(face, tip, base)`, with the face as it is now and two world points on the tool: its tip, and a point back along it (`base`) that
    stays in front of the face when the tool is fully in. The tip is followed in the face's own frame, so a face that moves is followed too.
    - `entered` is set when the tip crosses the face this step inside its outline (from the front, unless `from_front` is False), with the tool's
      line crossing the face inside the outline. It stays set while the tool straddles the face, crossing it inside the outline;
    - an entered tool whose crossing leaves the outline sideways while it still straddles the face latches `torn` (dragged out through the edge);
    - an entered tool no longer straddling the face is out again (`withdrawn` once the tip is back in front);
    - `depth_now` is how far the tip is behind the face; `inserted` is entered and at least `depth` deep now; `reached` latches the first time it is.
    `crossing` is where the tool's line crosses the face now, in the face's x, y, or None.

    Reads: the tip going in through the face, then on to depth. Blocks: a colliderless needle slid in along the face from beside the outline or past
    its edge; a shallow insertion credited as seated; a tool pulled out early (inserted is False again, and `reached` alone says it once got there).
    """

    def __init__(self, depth: float = 0.0, from_front: bool = True):
        self.depth, self.from_front = float(depth), from_front
        self.reset()

    def reset(self, env=None) -> None:
        self.entered = self.reached = self.torn = self.withdrawn = False
        self.prev, self.depth_now, self.crossing = None, 0.0, None

    def step(self, face: Face, tip, base) -> bool:
        rt, rb = face.local(tip), face.local(base)
        ht, hb = rt[2], rb[2]
        straddle = ht * hb < 0.0
        self.crossing = (rb + (rt - rb) * (hb / (hb - ht)))[:2] if straddle else None
        crossing_in = straddle and face.inside(self.crossing)
        if self.prev is not None:
            hp = self.prev[2]
            if hp * ht < 0.0 and (hp > 0.0 or not self.from_front) and face.inside(self.prev + (rt - self.prev) * (hp / (hp - ht))) and crossing_in:
                self.entered = True  # the tip went through the face, inside the outline, this step
        if self.entered and straddle and not crossing_in:
            self.entered, self.torn = False, True
        elif self.entered and not straddle:
            self.entered = False
            self.withdrawn = self.withdrawn or ht > 0.0
        self.prev = rt
        self.depth_now = float(-ht)
        self.reached = self.reached or self.inserted
        return self.inserted

    @property
    def inserted(self) -> bool:
        return self.entered and self.depth_now >= self.depth


# -- a strike, read at the struck object's surface, relative to it ----------------------------------------------------------------------------------


@dataclass
class Blow:
    """A blow that landed: the point's index, its speed in toward the object (m/s, relative to it) and where it was moving that fast (object frame)."""

    index: int
    speed: float
    where: np.ndarray


class Strike:
    """A strike, push or tap read as contact at the struck object's surface, at a speed relative to the object.

    Each step: `step(local, dt, zone, gap, normal, skip=None)`, with `local` the striking points in the object's frame now (`part.to_local(tips)`, or a
    held tool's head points), so their change between steps is their velocity relative to the object, whatever the object does. For each point:
    - in `zone(q)` (where a blow counts), its speed in toward the object, `-(v . normal(q))`, is read; `fastest` keeps the peak, contact or not;
    - one moving in at `v_min` or more must reach the object, `gap(q)` (its distance to the object's real surface) at most `contact`, that step or the
      next, to land; a blow lands at the faster of the two steps' speeds;
    - `skip` (a mask, as `hand_mask("left")`) drops a hand's points this step: a hand holding the object moves it, and does not strike it.
    `step` returns the blows that landed this step; `blows` keeps them all and `hardest` the fastest landed. Take a dent from `hardest` with a Latch.

    Reads: a finger or tool moving in against the object, at its surface. Blocks: the object carried, swung or rocked by a hand holding it (no relative
    speed, and the holding hand skipped); a punch pulled short; the object jerked toward a hovering finger that it never reaches; a fast swipe past the
    edge that never touches it (`gap` to the real surface, not to an endless wall); a tool carried along with the object.
    """

    def __init__(self, v_min: float, contact: float = 0.005):
        self.v_min, self.contact = float(v_min), float(contact)
        self.reset()

    def reset(self, env=None) -> None:
        self.prev, self.pending, self.blows, self.fastest, self.hardest = None, {}, [], 0.0, 0.0

    def step(self, local, dt: float, zone: Callable, gap: Callable, normal: Callable, skip=None) -> list[Blow]:
        loc = np.atleast_2d(np.asarray(local, float))
        landed: list[Blow] = []
        if self.prev is not None and self.prev.shape == loc.shape:
            vel = (loc - self.prev) / dt
            carry: dict[int, Blow | None] = {}
            for i, (q, v) in enumerate(zip(loc, vel)):
                if skip is not None and skip[i]:
                    continue
                blow = None
                if zone(q):
                    blow = Blow(i, -float(v @ np.asarray(normal(q), float)), q.copy())
                    self.fastest = max(self.fastest, blow.speed)
                pend = self.pending.get(i)
                g = gap(q) if (blow is not None or pend is not None) else math.inf
                fast = blow if blow is not None and blow.speed >= self.v_min else None
                if g <= self.contact:
                    hit = max([b for b in (pend, fast) if b is not None], key=lambda b: b.speed, default=None)
                    if hit is not None:
                        landed.append(hit)
                        self.hardest = max(self.hardest, hit.speed)
                    carry[i] = None
                else:
                    carry[i] = fast
            self.pending = {i: b for i, b in carry.items() if b is not None}
        self.prev = loc.copy()
        self.blows.extend(landed)
        return landed


def ring_zone(r_lo: float, r_hi: float, z_lo: float, z_hi: float) -> Callable:
    """A zone for `Strike`: a ring round a part's z axis, from radius r_lo to r_hi and height z_lo to z_hi in its frame (a rim, a seam, a band)."""
    return lambda q: bool(r_lo <= math.hypot(q[0], q[1]) <= r_hi and z_lo <= q[2] <= z_hi)


@dataclass(frozen=True)
class Cylinder:
    """A round part's outer surface about its z axis, from z0 to z1, in its own frame, for `Strike`: `radius` a number, or a function of z (a taper)."""

    radius: float | Callable
    z0: float
    z1: float

    def r(self, z: float) -> float:
        return float(self.radius(z)) if callable(self.radius) else float(self.radius)

    def gap(self, q) -> float:
        """Distance from q to the real surface: radial beside it, to the top or bottom edge beyond its ends, negative inside the wall's line."""
        dr = math.hypot(q[0], q[1]) - self.r(float(np.clip(q[2], self.z0, self.z1)))
        dz = max(float(q[2]) - self.z1, self.z0 - float(q[2]))
        return math.hypot(max(dr, 0.0), dz) if dz > 0.0 else dr

    def normal(self, q) -> np.ndarray:
        """Out from the axis, level."""
        n = np.array([q[0], q[1], 0.0], float)
        return n / max(float(np.linalg.norm(n)), 1e-9)


# -- partial credit ---------------------------------------------------------------------------------------------------------------------------------


class Ladder:
    """A partial-credit table that rises with each real milestone: rungs (name, value) from the lowest to the top, values strictly rising, the top 1.0.

    `score(reached, failed=False)` gives the highest rung reached (a set of names, or a mapping name -> bool), 0 if none, and 0 whenever `failed`
    (a latched failure). Give each count its own rung (1 turn, 2 turns, ...) so that every further real step earns something.
    A table with a flat stretch, a falling value or a top that is not 1.0 raises when it is built.
    """

    def __init__(self, *rungs: tuple[str, float]):
        names = [n for n, _ in rungs]
        values = [float(v) for _, v in rungs]
        if not rungs or len(set(names)) != len(names):
            raise ValueError("a ladder needs rungs with distinct names")
        if values[0] <= 0.0 or any(b <= a for a, b in zip(values, values[1:])):
            raise ValueError(f"rung values must rise strictly from above 0: {values}")
        if abs(values[-1] - 1.0) > 1e-12:
            raise ValueError(f"the top rung is 1.0, not {values[-1]}")
        self.rungs = tuple(zip(names, values))

    def score(self, reached, failed: bool = False) -> float:
        if failed:
            return 0.0
        got = {k for k, v in reached.items() if v} if isinstance(reached, Mapping) else set(reached)
        unknown = got - {n for n, _ in self.rungs}
        if unknown:
            raise KeyError(f"no rung named {sorted(unknown)}")
        for name, value in reversed(self.rungs):
            if name in got:
                return value
        return 0.0


def check_ladder(score: Callable, nothing, milestones: Sequence = (), routes: Sequence = (), show: bool = True) -> list[str]:
    """Check a scorer's partial credit on hand-built cases; returns what is wrong (empty if nothing).

    score(case) -> float runs the task's own scoring on a case (a record, or a list of per-step states fed to its Trial).
    nothing: the do-nothing case, which must score 0.
    milestones: (name, case) in order, each adding one more real milestone: each must score more than the last, and the last 1.0.
    routes: (name, case, cap): lazy routes, each scoring no more than its cap (0 unless the text gives it credit).
    """
    problems, rows = [], []
    s0 = float(score(nothing))
    rows.append(("doing nothing", s0, s0 == 0.0))
    if s0 != 0.0:
        problems.append(f"doing nothing scores {s0}, not 0")
    prev = s0
    for name, case in milestones:
        s = float(score(case))
        ok = s > prev + 1e-12
        rows.append((f"milestone: {name}", s, ok))
        if not ok:
            problems.append(f"milestone '{name}' scores {s}, no more than the step before it ({prev})")
        prev = s
    if milestones and abs(prev - 1.0) > 1e-9:
        problems.append(f"the last milestone scores {prev}, not 1.0")
    for name, case, cap in routes:
        s = float(score(case))
        ok = s <= cap + 1e-9
        rows.append((f"lazy route: {name} (at most {cap})", s, ok))
        if not ok:
            problems.append(f"lazy route '{name}' scores {s}, above its cap {cap}")
    if show:
        for label, s, ok in rows:
            print(f"{'ok ' if ok else 'BAD'} {s:.4f}  {label}")
    return problems


# -- hand-built states, for check_score.py -----------------------------------------------------------------------------------------------------------


class Pose:
    """A stand-in for mechkit's PartState on hand-built states: pos, rotation(), to_local(), to_world(), lin_vel, ang_vel, joint(), deg()."""

    def __init__(self, pos=(0.0, 0.0, 0.0), R=None, lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0), joints: dict | None = None):
        self.pos = np.asarray(pos, float)
        self.R = np.eye(3) if R is None else np.asarray(R, float)
        self.lin_vel, self.ang_vel = np.asarray(lin_vel, float), np.asarray(ang_vel, float)
        self.joints = dict(joints or {})

    def rotation(self) -> np.ndarray:
        return self.R

    def to_local(self, points) -> np.ndarray:
        return (np.atleast_2d(points) - self.pos) @ self.R

    def to_world(self, points) -> np.ndarray:
        return np.atleast_2d(points) @ self.R.T + self.pos

    def tilt_deg(self) -> float:
        return float(np.degrees(np.arccos(np.clip(self.R[2, 2], -1.0, 1.0))))

    def joint(self, link: str) -> float:
        return self.joints[f"{link}_joint"] if f"{link}_joint" in self.joints else self.joints[link]

    def deg(self, link: str) -> float:
        return math.degrees(self.joint(link))


DOWN = np.diag([1.0, -1.0, -1.0])  # a hand pointing straight down, its jaws closing along x


def hand(mid, R=None, opening: float = 0.0) -> np.ndarray:
    """One hand's 12 fingertip points laid out as mechkit's: `mid` the point midway between the two fingertips, `R` the hand's rotation (columns:
    the direction the jaws close along, across the fingers' width, and the way the fingers point; default `DOWN`), `opening` each pad's distance from
    the middle (half the jaw gap)."""
    R = DOWN if R is None else np.asarray(R, float)
    origin = np.asarray(mid, float) - R @ np.array([0.0, 0.0, FINGER_DZ[0]])
    return np.array([origin + R @ np.array([sx * (opening + PAD_INSET), dy, dz]) for sx in (-1.0, 1.0) for dy in FINGER_DY for dz in FINGER_DZ])


PARKED = {"left": (0.0, 2.0, 2.0), "right": (0.0, -2.0, 2.0)}  # far from anything


def tips(left=None, right=None) -> np.ndarray:
    """The 24 points from two hands' 12 (from `hand`); a hand not given is parked far away, open."""
    return np.vstack([np.asarray(left, float) if left is not None else hand(PARKED["left"], opening=0.03),
                      np.asarray(right, float) if right is not None else hand(PARKED["right"], opening=0.03)])


# -- lazy routes: cheat modes for a demo script ------------------------------------------------------------------------------------------------------
# Each gate needs a check that tries the lazy route and scores no more than the text says. Try them first on hand-built states in check_score.py;
# run the few that turn on physics as demo modes. LAZY_ROUTES names each route, what it does, the gate that must stop it, and the score it may reach
# by default (0: no unintended strategy scores above 0; give a route a higher cap only where the text credits a real step it makes).

LAZY_ROUTES = {
    "none": ("does nothing", "every rung: doing nothing scores 0", 0.0),
    "hover": ("an open hand put round the part, the jaws never closed", "held(): jaws closed on the part, pads on both sides", 0.0),
    "carry": ("the part or the whole assembly carried by a hand holding it, with no strike or turn made", "Strike: speed relative to the object, holding hand skipped", 0.0),
    "rock": ("the part rocked back and forth instead of turned", "NetRotation: the net signed turn against what it turns in", 0.0),
    "nudge": ("the part pushed along by closed fingers instead of picked up", "held(): a grasp, and the part off its rest", 0.0),
    "sideways": ("a tool with no collider slid in along the face from beside the outline", "PathInsert: the tip crosses the face inside the outline", 0.0),
    "early_out": ("the tool or part pulled out before the step is done", "PathInsert.inserted now, HoldThroughout, EndHold", 0.0),
    "wrong_hand": ("the other hand does the job, or pins the part at the end", "held(hand=...), hands_clear over all 24 points", 0.0),
    "in_stand": ("the tool's effect used while it lies in its stand, the work moved to it", "held(resting=on_support(...)): the tool in a hand and off its stand", 0.0),
}
LAZY_MODES = tuple(LAZY_ROUTES)


def route_table(scores: Mapping[str, float], caps: Mapping[str, float] | None = None, what: Mapping[str, str] | None = None) -> tuple[str, list[str]]:
    """The README's failure table from the lazy routes' scores (mode -> score), and the routes that scored above their caps."""
    caps, what = dict(caps or {}), dict(what or {})
    rows = ["| Mode | What it does | Gate that stops it | Score | Allowed |", "|---|---|---|---|---|"]
    over = []
    for mode, s in scores.items():
        did, gate, cap = LAZY_ROUTES.get(mode, (what.get(mode, ""), "", 0.0))
        cap = caps.get(mode, cap)
        rows.append(f"| {mode} | {what.get(mode, did)} | {gate} | {s:.2f} | {cap:.2f} |")
        if s > cap + 1e-9:
            over.append(mode)
    return "\n".join(rows), over


def lazy_hands(ctx):
    """(the first arm, the second arm) for a script written as `a, b = gatekit.lazy_hands(ctx)`: left and right, swapped in mode "wrong_hand"."""
    return (ctx.right, ctx.left) if ctx.mode == "wrong_hand" else (ctx.left, ctx.right)


def _v(p) -> np.ndarray:
    return np.asarray(p, dtype=float)


def lazy_hover(arm, at, grip, back: float = 0.06, dwell: float = 2.0, t: float = 1.5):
    """Mode "hover": the open hand brought round the part, its TCP at `at`, held there `dwell` s without closing, then backed off."""
    a = arm.approach_axis(grip)
    yield from arm.open(t=0.3, label="hover: open the hand")
    yield from arm.move(_v(at) - back * a + np.array([0.0, 0.0, 0.02]), grip=grip, t=t, label="hover: behind the part")
    yield from arm.move(_v(at), t=1.0, label="hover: open fingers either side of it")
    yield from arm.wait(dwell)
    yield from arm.move(_v(at) - back * a, t=0.8, label="hover: back off")


def lazy_carry(arm, at, grip, width: float, shift=(0.0, 0.06, 0.0), lift: float = 0.01, times: int = 2, t: float = 0.3, force: float = 30.0):
    """Mode "carry": take the part at `at`, lift it `lift`, carry it by `shift` and back `times` times (0.06 m in 0.3 s: 0.2 m/s), set it down, let go."""
    a = arm.approach_axis(grip)
    yield from arm.open(t=0.3, label="carry: open the hand")
    yield from arm.move(_v(at) - 0.09 * a + np.array([0.0, 0.0, 0.02]), grip=grip, t=1.5, label="carry: before the part")
    yield from arm.move(_v(at), t=1.2, label="carry: fingers either side of it")
    yield from arm.close(width, force, label="carry: grip it")
    yield from arm.move(arm.at() + np.array([0.0, 0.0, lift]), t=0.5, label="carry: lift it")
    for k in range(times):
        yield from arm.move(arm.at() + _v(shift), t=t, label=f"carry: away ({k + 1})")
        yield from arm.move(arm.at() - _v(shift), t=t, label=f"carry: back ({k + 1})")
    yield from arm.move(arm.at() - np.array([0.0, 0.0, lift]), t=0.5, label="carry: set it down")
    yield from arm.open(label="carry: let go")
    yield from arm.move(arm.at() - 0.09 * a, t=0.8, label="carry: back off")


def lazy_rock(arm, at, grip, width: float, deg: float = 12.0, axis=(1.0, 0.0, 0.0), times: int = 3, t: float = 0.2, force: float = 30.0):
    """Mode "rock": take the part at `at` and rock it `deg` each way about `axis` through the TCP, `times` times, then let go: no net turn."""
    a = arm.approach_axis(grip)
    yield from arm.open(t=0.3, label="rock: open the hand")
    yield from arm.move(_v(at) - 0.09 * a + np.array([0.0, 0.0, 0.02]), grip=grip, t=1.5, label="rock: before the part")
    yield from arm.move(_v(at), t=1.2, label="rock: fingers either side of it")
    yield from arm.close(width, force, label="rock: grip it")
    for k in range(times):
        yield from arm.turn(deg, axis=axis, t=t, label=f"rock: one way ({k + 1})")
        yield from arm.turn(-2 * deg, axis=axis, t=2 * t, label=f"rock: the other ({k + 1})")
        yield from arm.turn(deg, axis=axis, t=t, label=f"rock: back ({k + 1})")
    yield from arm.open(label="rock: let go")
    yield from arm.move(arm.at() - 0.09 * a, t=0.8, label="rock: back off")


def lazy_nudge(arm, frm, to, grip, t: float = 1.5):
    """Mode "nudge": closed fingers brought to `frm` (from 4 cm above it) and pushed to `to`, then lifted away: the part shoved, never grasped."""
    yield from arm.close(t=0.4, label="nudge: close the fingers")
    yield from arm.move(_v(frm) + np.array([0.0, 0.0, 0.04]), grip=grip, t=2.0, label="nudge: above the start")
    yield from arm.move(_v(frm), t=0.8, label="nudge: at the part")
    yield from arm.move(_v(to), t=t, label="nudge: push it")
    yield from arm.move(arm.at() + np.array([0.0, 0.0, 0.06]), t=0.8, label="nudge: hand away")


def lazy_sideways(arm, beside, inside, read=None, t: float = 1.5):
    """Mode "sideways": a held tool's tip brought to `beside` (just outside the face's outline, already at depth) and slid along the face to `inside`.
    With `read` (a function giving the tip's world point now), closed loop on the tip, as a needle demo is; without it, `beside` and `inside` are
    TCP positions."""
    beside, inside = _v(beside), _v(inside)
    if read is None:
        yield from arm.move(beside, t=2.0, label="sideways: beside the outline")
        yield from arm.move(inside, t=t, label="sideways: slid in along the face")
        return
    yield from arm.servo(beside, tol=0.0015, timeout=2.0, read=read, label="sideways: the tip beside the outline")
    yield from arm.servo(lambda s: beside + (inside - beside) * min(s / t, 1.0), tol=0.0015, timeout=t + 1.0, read=read, label="sideways: slid in along the face")


def lazy_early_out(arm, dist: float = 0.06, direction=None, t: float = 0.6):
    """Mode "early_out": called where the script would go on (turn, fire, seat, hold), it pulls the tool back `dist` along `direction` (default: back
    along the grip's axis) at once."""
    d = -arm.approach_axis() if direction is None else _v(direction) / np.linalg.norm(direction)
    yield from arm.move(arm.at() + dist * d, t=t, label="early_out: pulled out")
