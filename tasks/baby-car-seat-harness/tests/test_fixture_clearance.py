"""Clearance regressions for the front adjustment tab and the visible free tail."""

import numpy as np

import harness_fit as hf
import scene_assets as sa


def test_tail_and_gripper_pad_envelope_stay_in_front_of_carrier_rim():
    # The entire slanted foot wall has x >= this support-plane bound.
    rim_min_x = sa.X0 + sa.C * sa.SHELL_U[0] - sa.S * sa.SHELL_RIM
    # Straight-down pads are 36 mm wide in x. Include both ends of the
    # complete physical pull, rather than only testing the tab centre.
    pull_x = np.array([sa.ADJUSTER_START[0], sa.ADJUSTER_START[0] - hf.SHOULDER_SLACK])
    pad_max_x = float(pull_x.max()) + 0.018
    assert rim_min_x - pad_max_x >= 0.010
    assert sa.ADJUSTER_START[2] - 0.013 >= 0.050


def test_visible_tail_exits_above_the_front_rim():
    rim_max_z = sa.Z0 + sa.S * (sa.SHELL_U[0] + sa.SHELL_WALL) + sa.C * sa.SHELL_RIM
    assert sa.ADJUSTER_EXIT[2] - rim_max_z >= 0.010


def test_open_pockets_have_enough_material_but_do_not_start_snug():
    for side in (1, -1):
        tongue = sa.to_seat(sa.tongue_start(side))[0]
        available = hf.material_feed_limit(side, tongue, sa.material_budget(side))
        assert 0.008 <= available <= 0.010
        assert available > 2 * hf.FOLD_OVERLAP
