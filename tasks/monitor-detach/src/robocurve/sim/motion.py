"""Smooth arm targets for scripted policies: min-jerk moves in joint space or Cartesian space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from robocurve.sim.arm_math import Kinematics, Side, rotation_error

Vec = npt.NDArray[np.float64]


TIME_EPS = 1e-6  # control ticks land on float times; a move ending "now" is done [s]


def min_jerk(s: float) -> float:
    """Min-jerk time scaling for s in [0, 1]."""
    s = float(np.clip(s, 0.0, 1.0))
    return s**3 * (10 - 15 * s + 6 * s * s)


def slerp_rot(r0: Vec, r1: Vec, s: float) -> Vec:
    """Rotation a fraction s of the way from r0 to r1 (geodesic)."""
    vel = rotation_error(r1, r0)
    angle = np.linalg.norm(vel)
    if angle < 1e-9:
        return r0.copy()
    axis = vel / angle
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    step = np.eye(3) + np.sin(s * angle) * k + (1 - np.cos(s * angle)) * (k @ k)
    return step @ r0


@dataclass
class _Move:
    kind: str  # "joint" or "cart"
    t0: float
    duration: float
    start: tuple
    end: tuple


class ArmTrack:
    """Target generator for one arm. Joint moves interpolate q; Cartesian moves interpolate the TCP.

    ``pos``, ``rot`` and ``q`` describe the nominal (undeflected) arm. With a joint ``offset`` set,
    the command is ``q + offset``: the arm repeats the nominal motion from wherever it really is.
    """

    def __init__(self, side: Side, kin: Kinematics, q0: Vec) -> None:
        self.side = side
        self.kin = kin
        self.q = np.asarray(q0, dtype=float).copy()
        self.pos, self.rot = kin.fk(side, self.q)
        self.grip = 0.0  # 0 = open, 1 = closed
        self.offset: Vec | None = None
        self._move: _Move | None = None

    def command(self) -> Vec:
        """Joint target to send to the arm."""
        return self.q if self.offset is None else self.q + self.offset

    def move_joint(self, t: float, q_target: Vec, duration: float) -> None:
        self.update(t)  # settle a move that ended this tick, so the new one starts from its endpoint
        self._move = _Move("joint", t, duration, (self.q.copy(),), (np.asarray(q_target, dtype=float),))

    def move_cart(self, t: float, pos: Vec, rot: Vec, duration: float) -> None:
        self.update(t)  # settle a move that ended this tick, so the new one starts from its endpoint
        self._move = _Move("cart", t, duration, (self.pos.copy(), self.rot.copy()), (np.asarray(pos), np.asarray(rot)))

    def done(self, t: float) -> bool:
        return self._move is None or t >= self._move.t0 + self._move.duration - TIME_EPS

    def update(self, t: float) -> Vec:
        """Advance to time t and return the arm joint target (IK for Cartesian moves)."""
        move = self._move
        if move is None:
            return self.q
        s = min_jerk((t - move.t0) / max(move.duration, 1e-6))
        if move.kind == "joint":
            self.q = move.start[0] + s * (move.end[0] - move.start[0])
            self.pos, self.rot = self.kin.fk(self.side, self.q)
        else:
            self.pos = move.start[0] + s * (move.end[0] - move.start[0])
            self.rot = slerp_rot(move.start[1], move.end[1], s)
            self.q = self.kin.solve(self.side, self.pos, self.rot, q_init=self.q, iters=60).q
        if t >= move.t0 + move.duration - TIME_EPS:
            self._move = None
        return self.q
