"""stemjoint: how an apple hangs on its spur in the practice rig, and when it comes off.

The rig: the spur's tip holds a disc magnet; the apple's stem ends in a steel cap that sits flat on it; the stem's other
end is a peg held in a socket in the apple by a weaker magnet. In the simulation each apple is a free rigid body held at
the cap (the seat) by a joint spring that the task file applies every physics step:

- stiff in travel; in bending as stiff as the flexible stem (pre-bent so the apple hangs level), so a lift or a sideways
  bend of about 60-90 degrees loads the seat to its limit, while a pull lines the apple up with the stem;
- in twist, only up to the friction torque of the cap on the magnet; beyond it the apple turns on the seat.

The spring's force and moment are the loads on the seat. Every physics step they are checked against two limits, and
the first threshold crossing releases the spring and the apple is free:

- the cap parts from the magnet when the tension along the stem plus the bending moment divided by the magnet's radius
  reaches ``SEAT_PULL`` (it tips over the magnet's edge): a clean pick, stem on the apple;
- the peg pulls out of the apple when the tension reaches ``PEG_PULL``: a stem pull, the stem stays on the spur.

Pure NumPy, no simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

SEAT_PULL = 40.0           # N: the cap's hold on the spur's magnet (session-check target 40 +- 3 N)
SEAT_RADIUS = 0.00635      # m: the magnet's radius (1/2 in disc): the cap tips about its edge
PEG_PULL = 24.0            # N: the peg's hold in the apple's socket (session-check target 24 +- 2 N)
CAP_FRICTION = 0.25        # friction of the steel cap on the nickel-plated magnet
K_LIN = 2.0e4              # N/m: the seat's stiffness along any direction (integrated implicitly)
C_LIN = 6.0                # N s/m
K_BEND = 0.21              # N m/rad: the printed TPU stem bends (E I / L, 66 MPa x 63.617 mm^4 / 20 mm)
K_TWIST = 0.14             # N m/rad (G J / L of the TPU stem), up to the cap's friction torque
C_ROT = 0.004              # N m s/rad


def tip_moment() -> float:
    """The bending moment that tips the cap off the magnet with no tension: about 0.25 N m."""
    return SEAT_PULL * SEAT_RADIUS


def twist_limit(tension: float) -> float:
    """The torque at which the cap turns on the magnet: friction over the face (mean radius 2/3 r) under the clamp."""
    return CAP_FRICTION * max(SEAT_PULL - tension, 0.0) * 2.0 / 3.0 * SEAT_RADIUS


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def rotvec(r: np.ndarray) -> np.ndarray:
    """The rotation vector (axis times angle, rad) of a rotation matrix."""
    c = min(1.0, max(-1.0, (np.trace(r) - 1.0) / 2.0))
    angle = math.acos(c)
    w = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if angle < 1e-7:
        return w / 2.0
    if math.pi - angle < 1e-4:
        k = int(np.argmax(np.diag(r)))
        axis = r[:, k] + np.eye(3)[k]
        return axis / np.linalg.norm(axis) * angle
    return w / (2.0 * math.sin(angle)) * angle


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def load_ratios(loads: dict) -> tuple[float, float]:
    tension = loads["tension"]
    return (max(tension, 0.0) + loads["bending"] / SEAT_RADIUS) / SEAT_PULL, tension / PEG_PULL


def first_crossing(previous, current) -> str | None:
    """Interpolate both overload ratios across the physics sample; an exact tie is a peg pull."""
    fractions = [(1.0 - old) / (new - old) if new >= 1.0 and old < 1.0 else math.inf
                 for old, new in zip(previous, current)]
    if min(fractions) == math.inf:
        return None
    return "seat" if fractions[0] < fractions[1] - 1e-12 else "stem"


def part_of(loads: dict) -> str | None:
    """First threshold on a proportional ramp from zero to these loads."""
    return first_crossing((0.0, 0.0), load_ratios(loads))


@dataclass
class Joint:
    """One apple's seat. Frames: the tree's (``rot0``, ``point`` are in it) and the apple's (``anchor``: the seat, from
    the apple's centre of mass; the stem runs along the apple's +x, from the apple to the spur)."""

    name: str
    anchor: np.ndarray             # apple frame: centre of mass -> the cap on the magnet
    point: np.ndarray              # tree frame: the magnet's face as the apple hangs
    rot0: np.ndarray               # tree frame: the apple's rotation as it hangs (turned as the cap slips in twist)
    mass: float
    inertia: float                 # kg m^2 (the apple taken as a uniform sphere)
    preload: np.ndarray = field(default_factory=lambda: np.zeros(3))  # tree frame: the moment that holds the apple up
    broken: str | None = None      # None, "seat" (clean pick) or "stem" (stem pull)
    peak: dict = field(default_factory=dict)
    previous_ratios: tuple[float, float] = (0.0, 0.0)

    def settle(self, gravity: np.ndarray, tree_rot: np.ndarray) -> None:
        """Hold the apple still where it was placed: the seat carries its weight's moment from the start."""
        r = tree_rot @ self.rot0 @ self.anchor
        self.preload = tree_rot.T @ np.cross(r, self.mass * gravity)
        self.broken = None
        self.previous_ratios = (0.0, 0.0)
        self.peak = {"tension": 0.0, "bending": 0.0, "twist": 0.0, "seat_ratio": 0.0, "stem_ratio": 0.0}

    def wrench(self, apple_pos, apple_rot, apple_v, apple_w, tree_pos, tree_rot, tree_com, tree_v, tree_w, dt: float):
        """The seat's force (at the cap) and moment on the apple, world frame, and the loads it carries."""
        r = apple_rot @ self.anchor
        p_apple = apple_pos + r
        p_tree = tree_pos + tree_rot @ self.point
        e = p_apple - p_tree
        v = apple_v + np.cross(apple_w, r) - (tree_v + np.cross(tree_w, p_tree - tree_com))
        sr = skew(r)
        inv_mass = np.eye(3) / self.mass - sr @ sr / self.inertia
        rhs = K_LIN * e + (K_LIN * dt + C_LIN) * v
        force = -np.linalg.solve(np.eye(3) + (K_LIN * dt * dt + C_LIN * dt) * inv_mass, rhs)
        u = apple_rot @ np.array([1.0, 0.0, 0.0])   # along the stem, apple to spur
        tension = float(force @ u)
        phi = rotvec(apple_rot @ (tree_rot @ self.rot0).T)
        twist = float(phi @ u)
        limit = twist_limit(tension)
        if abs(K_TWIST * twist) > limit:   # the cap turns on the magnet: move its rest so the twist torque stays at the limit
            slip = twist - math.copysign(limit / K_TWIST, twist)
            self.rot0 = self.rot0 @ rot_x(slip)
            twist -= slip
            phi = rotvec(apple_rot @ (tree_rot @ self.rot0).T)
        bend = phi - float(phi @ u) * u
        w_rel = apple_w - tree_w
        moment = (-K_BEND * bend - K_TWIST * twist * u - C_ROT * w_rel + tree_rot @ self.preload)
        m_twist = float(moment @ u)
        loads = {"tension": tension, "bending": float(np.linalg.norm(moment - m_twist * u)), "twist": abs(m_twist),
                 "shear": float(np.linalg.norm(force - tension * u))}
        return force, moment, p_tree, loads

    def check(self, loads: dict) -> str | None:
        for key in ("tension", "bending", "twist"):
            self.peak[key] = max(self.peak.get(key, 0.0), loads[key])
        self.peak["seat_ratio"] = max(self.peak["seat_ratio"],
                                      (max(loads["tension"], 0.0) + loads["bending"] / SEAT_RADIUS) / SEAT_PULL)
        self.peak["stem_ratio"] = max(self.peak["stem_ratio"], loads["tension"] / PEG_PULL)
        current = load_ratios(loads)
        part = first_crossing(self.previous_ratios, current)
        self.previous_ratios = current
        return part
