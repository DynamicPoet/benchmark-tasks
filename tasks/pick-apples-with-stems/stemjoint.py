"""stemjoint: how an apple hangs on its spur, and when the stem, the abscission joint or the spur gives way.

Each apple is a free rigid body held to the tree at its abscission joint (where the stem meets the spur) by a joint spring
that the task file applies every physics step: stiff in every direction of travel (the stem does not stretch more than
about a millimetre before it tears), and soft in bending and twisting, as a 20 mm apple stem is. The spring's force and
moment are the loads on the joint. They are checked against three limits every physics step:

- the abscission joint snaps when its bending and twisting moment together reach ``ABSCISSION_MOMENT``, or its tension or
  shear reaches ``ABSCISSION_FORCE``: a clean pick, stem on the apple, spur on the tree;
- the stem tears out of the fruit (a stem pull) when the tension along it reaches ``STEM_TENSION``;
- the spur breaks off the trunk when the moment at its base reaches ``SPUR_MOMENT`` or its tension ``SPUR_TENSION``.

When a limit is reached the spring is no longer applied: from then on the apple is a free body. Sources for the numbers are
in the task's README.

Pure NumPy, no simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

ABSCISSION_MOMENT = 0.26   # N m: about the twisting torque that detached 95% of hand-picked apples (Dickinson 2022)
ABSCISSION_FORCE = 33.0    # N: an obtuse-angled branch-stem joint in a straight pull, 33.19 +- 7.48 N (Bu et al. 2020)
STEM_TENSION = 25.0        # N: the stem pulled out of the fruit, 24.78 +- 0.48 N (Hussain et al. 2022)
SPUR_MOMENT = 0.80         # N m: assumed; a spur outlasts the abscission joint unless levered hard
SPUR_TENSION = 45.0        # N: assumed; above the stem's tension limit, so a straight pull tears the stem first

BEND_TURN = math.radians(90.0)   # the abscission joint snaps after about this much bending from where it hangs, ...
TWIST_TURN = math.radians(90.0)  # ... or this much twist
K_LIN = 2.0e4                    # N/m: the joint's stiffness along any direction (1.25 mm of stretch at the stem's limit)
C_LIN = 6.0                      # N s/m
C_ROT = 0.002                    # N m s/rad


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def rotvec(r: np.ndarray) -> np.ndarray:
    """The rotation vector (axis times angle, rad) of a rotation matrix."""
    c = (np.trace(r) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    angle = math.acos(c)
    w = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if angle < 1e-7:
        return w / 2.0
    if math.pi - angle < 1e-4:
        k = int(np.argmax(np.diag(r)))
        axis = r[:, k] + np.eye(3)[k]
        return axis / np.linalg.norm(axis) * angle
    return w / (2.0 * math.sin(angle)) * angle


@dataclass
class Joint:
    """One apple's attachment. Frames: the tree's (``rot0``, ``point``, ``spur_dir`` are in it) and the apple's
    (``anchor``: the abscission joint from the apple's centre of mass)."""

    name: str
    anchor: np.ndarray             # apple frame: centre of mass -> abscission joint
    point: np.ndarray              # tree frame: the abscission joint as the apple hangs
    rot0: np.ndarray               # tree frame: the apple's rotation as it hangs
    spur_dir: np.ndarray           # tree frame: the spur's outward direction (unit)
    spur_len: float
    mass: float
    inertia: float                 # kg m^2 (the apple taken as a uniform sphere)
    preload: np.ndarray = field(default_factory=lambda: np.zeros(3))  # tree frame: the moment that holds the apple up
    k_bend: float = 0.0
    k_twist: float = 0.0
    broken: str | None = None      # None, "abscission", "stem" or "spur"
    peak: dict = field(default_factory=dict)

    def settle(self, gravity: np.ndarray, tree_rot: np.ndarray) -> None:
        """Pre-bend the joint so the apple hangs still where it was placed, and set its stiffness so that the abscission
        joint snaps after BEND_TURN of bending beyond that or TWIST_TURN of twist."""
        r = tree_rot @ self.rot0 @ self.anchor
        weight_moment = np.cross(r, self.mass * gravity)
        self.preload = tree_rot.T @ weight_moment
        self.k_bend = (ABSCISSION_MOMENT + float(np.linalg.norm(weight_moment))) / BEND_TURN
        self.k_twist = ABSCISSION_MOMENT / TWIST_TURN
        self.broken = None
        self.peak = {"moment": 0.0, "tension": 0.0, "shear": 0.0, "spur_moment": 0.0, "stem": 0.0}

    def wrench(self, apple_pos, apple_rot, apple_v, apple_w, tree_pos, tree_rot, tree_com, tree_v, tree_w, dt: float):
        """The joint's force (at the abscission joint) and moment on the apple, world frame, and the loads it carries.
        The force is integrated implicitly (backward Euler with the apple's own inertia), so it stays stable however stiff."""
        r = apple_rot @ self.anchor
        p_apple = apple_pos + r
        p_tree = tree_pos + tree_rot @ self.point
        e = p_apple - p_tree
        v = apple_v + np.cross(apple_w, r) - (tree_v + np.cross(tree_w, p_tree - tree_com))
        sr = skew(r)
        inv_mass = np.eye(3) / self.mass - sr @ sr / self.inertia
        rhs = K_LIN * e + (K_LIN * dt + C_LIN) * v
        force = -np.linalg.solve(np.eye(3) + (K_LIN * dt * dt + C_LIN * dt) * inv_mass, rhs)
        rest = tree_rot @ self.rot0
        phi = rotvec(apple_rot @ rest.T)
        u = apple_rot @ np.array([1.0, 0.0, 0.0])   # along the stem, apple to spur
        twist = float(phi @ u) * u
        moment = -self.k_bend * (phi - twist) - self.k_twist * twist - C_ROT * (apple_w - tree_w) + tree_rot @ self.preload
        return force, moment, p_tree, self.loads(force, moment, u, tree_rot)

    def loads(self, force: np.ndarray, moment: np.ndarray, u: np.ndarray, tree_rot: np.ndarray) -> dict:
        tension = float(force @ u)
        shear = float(np.linalg.norm(force - tension * u))
        d = tree_rot @ self.spur_dir
        # the apple's pull on the spur (-force at the joint, -moment), taken about the spur's base
        spur_moment = -moment + np.cross(self.spur_len * d, -force)
        return {"moment": float(np.linalg.norm(moment)), "tension": tension, "shear": shear,
                "spur_moment": float(np.linalg.norm(spur_moment)), "spur_tension": float(-force @ d)}

    def check(self, loads: dict) -> str | None:
        """Which part gives way under these loads, if any: the one loaded furthest past its limit."""
        for key in ("moment", "tension", "shear", "spur_moment"):
            self.peak[key] = max(self.peak.get(key, 0.0), loads[key])
        ratio = {
            "abscission": max(loads["moment"] / ABSCISSION_MOMENT, loads["tension"] / ABSCISSION_FORCE,
                              loads["shear"] / ABSCISSION_FORCE),
            "stem": loads["tension"] / STEM_TENSION,
            "spur": max(loads["spur_moment"] / SPUR_MOMENT, loads["spur_tension"] / SPUR_TENSION),
        }
        part = max(ratio, key=ratio.get)
        return part if ratio[part] >= 1.0 else None
