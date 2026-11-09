import numpy as np
import pytest

from motion_checks import points_at_rest


def stationary(samples=21):
    return [np.array([0.1, 0.2, 0.3, 0.2, 0.3, 0.4]) for _ in range(samples)]


@pytest.mark.parametrize("samples", [0, 1, 20])
def test_requires_a_full_second(samples):
    assert not points_at_rest(stationary(samples), 0.005, 20)


def test_stationary_cutting_passes():
    assert points_at_rest(stationary(), 0.005, 20)


@pytest.mark.parametrize("endpoint", [0, 1])
def test_motion_and_return_within_window_fails(endpoint):
    history = stationary()
    history[10][endpoint * 3] += 0.02
    assert not points_at_rest(history, 0.005, 20)


def test_diagonal_motion_uses_spatial_distance():
    history = [np.full(6, 0.004 * i / 20) for i in range(21)]
    assert not points_at_rest(history, 0.005, 20)


def test_motion_on_opposite_sides_of_start_fails():
    history = stationary()
    history[5][0] += 0.004
    history[15][0] -= 0.004
    assert not points_at_rest(history, 0.005, 20)


@pytest.mark.parametrize("distance, expected", [(0.0049, True), (0.005, False), (0.0051, False)])
def test_threshold_is_strict(distance, expected):
    history = [np.zeros(6) for _ in range(21)]
    history[10][0] = distance
    assert points_at_rest(history, 0.005, 20) is expected


def test_small_motion_passes():
    history = stationary()
    history[5][:3] += [0.001, 0.001, 0.001]
    history[15][3:] += [0.001, 0.001, 0.001]
    assert points_at_rest(history, 0.005, 20)


def test_motion_before_window_does_not_count():
    history = [np.ones(6)] + stationary()
    assert points_at_rest(history, 0.005, 20)


def test_non_finite_position_fails():
    history = stationary()
    history[10][0] = np.nan
    assert not points_at_rest(history, 0.005, 20)
