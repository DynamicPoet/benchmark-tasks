"""Types and rotation helpers shared by the arm kinematics and the scripted policies (NumPy only).

Frames are the station frame (origin at the left arm base, z up). Gripper axes, in the flange
(``gripper`` link) frame: the fingers close along +x/-x and the tool points along +z.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt

Side = Literal["left", "right"]
Vec = npt.NDArray[np.float64]

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
FINGER_JOINTS = ("joint7", "joint8")
# The finger carriages cross over, so each pad sits on the far side of its body origin. At q = 0 the
# two pads touch on the centreline (closed); each pad moves outward with q, to a 9.4 cm gap at the limit.
FINGER_CLOSED = 0.0
FINGER_OPEN = 0.04695


def gripper_rotation(approach: Vec, jaw: Vec) -> Vec:
    """Flange rotation matrix (columns = flange axes in world) for a tool direction and jaw axis.

    Args:
        approach: Direction the gripper points (flange +z). Normalized internally.
        jaw: Direction the fingers close along (flange +x). Made orthogonal to ``approach``.

    Returns:
        3x3 rotation matrix.
    """
    z = np.asarray(approach, dtype=float)
    z = z / np.linalg.norm(z)
    x = np.asarray(jaw, dtype=float)
    x = x - z * (x @ z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def rot_y(angle: float) -> Vec:
    """Rotation matrix about world +y by ``angle`` [rad]."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotation_error(r_target: Vec, r_current: Vec) -> Vec:
    """Axis-angle vector (world frame) that rotates ``r_current`` onto ``r_target``."""
    r = r_target @ r_current.T
    # The quaternion's vector part, from the largest of its four components (robust up to 180 deg).
    squares = np.array([1 + r[0, 0] - r[1, 1] - r[2, 2], 1 - r[0, 0] + r[1, 1] - r[2, 2],
                        1 - r[0, 0] - r[1, 1] + r[2, 2], 1 + r[0, 0] + r[1, 1] + r[2, 2]])
    largest = int(np.argmax(squares))
    q = np.zeros(4)  # x, y, z, w
    q[largest] = np.sqrt(max(squares[largest], 0.0)) / 2
    off = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])  # 4 w (x, y, z)
    pairs = np.array([r[1, 0] + r[0, 1], r[0, 2] + r[2, 0], r[2, 1] + r[1, 2]])  # 4 (xy, xz, yz)
    if largest == 3:
        q[:3] = off / (4 * q[3])
    else:
        q[3] = off[largest] / (4 * q[largest])
        others = {0: ((1, 0), (2, 1)), 1: ((0, 0), (2, 2)), 2: ((0, 1), (1, 2))}[largest]
        for index, pair in others:
            q[index] = pairs[pair] / (4 * q[largest])
    if q[3] < 0:
        q = -q
    n = np.linalg.norm(q[:3])
    return np.zeros(3) if n < 1e-12 else q[:3] / n * 2 * np.arctan2(n, q[3])


@dataclass
class IkResult:
    """Outcome of one IK solve."""

    q: Vec
    """Arm joint angles (joint1..joint6) [rad]."""
    pos_err: float
    """Remaining TCP position error [m]."""
    rot_err: float
    """Remaining flange orientation error [rad]."""

    @property
    def ok(self) -> bool:
        return self.pos_err < 2e-3 and self.rot_err < 2e-2


class Kinematics(Protocol):
    """What the scripted policies need from a kinematic model of both arms."""

    def joint_range(self, side: Side) -> Vec: ...

    def fk(self, side: Side, q: Vec) -> tuple[Vec, Vec]: ...

    def static_torques(self, side: Side, q: Vec, force: Vec) -> Vec: ...

    def solve(self, side: Side, pos: Vec, rot: Vec, q_init: Vec | None = None) -> IkResult: ...

    def solve_best(self, side: Side, pos: Vec, rot: Vec, seeds: list[Vec] | None = None) -> IkResult: ...
