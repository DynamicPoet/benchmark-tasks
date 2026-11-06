"""The demo's kinematics on yam_arena's URDF: IK reaches what FK says."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("yam_arena.urdf")

from robocurve.arena.kinematics import ArenaKinematics  # noqa: E402
from robocurve.sim.arm_math import rotation_error  # noqa: E402


@pytest.fixture(scope="module")
def kin() -> ArenaKinematics:
    return ArenaKinematics()


@pytest.mark.parametrize("side", ["left", "right"])
def test_ik_recovers_a_reachable_pose(kin: ArenaKinematics, side: str) -> None:
    q = np.array([0.1, 1.2, 1.1, -0.5, 0.2, 0.3])
    pos, rot = kin.fk(side, q)
    result = kin.solve(side, pos, rot, q_init=q + 0.05)
    assert result.ok


def test_bases_are_61_cm_apart(kin: ArenaKinematics) -> None:
    q = np.zeros(6)
    assert np.allclose(kin.fk("left", q)[0] - kin.fk("right", q)[0], [0.0, 0.61, 0.0])


def test_rotation_error_is_the_axis_angle_between_rotations() -> None:
    angle = 2.5
    c, s = np.cos(angle), np.sin(angle)
    target = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(rotation_error(target, np.eye(3)), [0.0, 0.0, angle])
