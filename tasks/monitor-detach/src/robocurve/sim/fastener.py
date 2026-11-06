"""Fastener, latch and scene faults for the monitor-detach task (plain Python, no simulator).

While latched, a stiff 6-DoF spring-damper holds the panel at its bracket pose. The simulation
applies the returned wrench at the panel's centre of mass every physics step. The latch opens
once lever travel reaches the scene's threshold and then stays open.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

Vec = npt.NDArray[np.float64]
GRAVITY = 9.81


@dataclass(frozen=True)
class SceneSpec:
    """One scene variant: latch threshold and Arm A's grip after release."""

    name: str
    latch_threshold: float
    """Lever travel fraction (0-1) at which the latch opens."""
    grip_fade: bool = False
    """B1: Arm A's grip force ramps down once the latch opens."""
    fade_duration_s: float = 2.0
    """Time for the grip to ramp from nominal to its floor (Arm A alone slips after ~1.8 s)."""
    fade_floor: float = 0.1
    """Grip force at the end of the ramp, as a fraction of nominal."""

    def grip_scale(self, t_since_release: float | None) -> float:
        """Arm A's grip force multiplier at a time since the latch opened (None: still latched)."""
        if not self.grip_fade or t_since_release is None or t_since_release <= 0.0:
            return 1.0
        frac = min(t_since_release / self.fade_duration_s, 1.0)
        return 1.0 - (1.0 - self.fade_floor) * frac


SCENES: dict[str, SceneSpec] = {
    "A": SceneSpec("A", latch_threshold=0.80),
    "B1": SceneSpec("B1", latch_threshold=0.80, grip_fade=True),
    "B2": SceneSpec("B2", latch_threshold=0.20),
}


def quat_xyzw_to_matrix(q: Vec) -> Vec:
    """Rotation matrix from a unit quaternion (x, y, z, w)."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotvec(r: Vec) -> Vec:
    """Axis-angle vector of a rotation matrix."""
    cos = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos)
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / (2.0 * np.sin(angle))
    return axis * angle


@dataclass
class Fastener:
    """Latch state plus the spring-damper wrench that holds the panel while latched."""

    rest_pos: Vec
    rest_quat: Vec
    """Bracket pose of the panel (x, y, z, w)."""
    mass: float
    threshold: float
    k_lin: float = 2.0e4
    c_lin: float = 300.0
    k_rot: float = 100.0
    c_rot: float = 2.0
    is_open: bool = False
    opened_at: float | None = None
    load_share_at_open: float | None = None
    last_force: Vec = field(default_factory=lambda: np.zeros(3))
    _rest_rot: Vec = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rest_pos = np.asarray(self.rest_pos, dtype=float)
        self._rest_rot = quat_xyzw_to_matrix(np.asarray(self.rest_quat, dtype=float))

    @property
    def load_share(self) -> float:
        """Vertical force the fastener carries, as a share of the panel's weight."""
        return float(self.last_force[2] / (self.mass * GRAVITY))

    def update_latch(self, t: float, lever_travel_frac: float) -> bool:
        """Open the latch once travel reaches the threshold. Returns True on the opening step."""
        if self.is_open or lever_travel_frac < self.threshold:
            return False
        self.is_open = True
        self.opened_at = t
        self.load_share_at_open = self.load_share
        return True

    def wrench(self, pos: Vec, quat: Vec, lin_vel: Vec, ang_vel: Vec) -> tuple[Vec, Vec]:
        """World-frame force and torque at the panel's centre of mass for the current state."""
        if self.is_open:
            self.last_force = np.zeros(3)
            return np.zeros(3), np.zeros(3)
        force = -self.k_lin * (np.asarray(pos) - self.rest_pos) - self.c_lin * np.asarray(lin_vel)
        r_err = quat_xyzw_to_matrix(np.asarray(quat)) @ self._rest_rot.T
        torque = -self.k_rot * rotvec(r_err) - self.c_rot * np.asarray(ang_vel)
        self.last_force = force
        return force, torque
