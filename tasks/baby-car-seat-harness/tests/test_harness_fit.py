import math

import numpy as np
import pytest

from harness_fit import (CENTRE_COLUMN, CONTACT_SLICE, CONTROL_ROWS, SHOULDER_SLICE, SHOULDER_SLACK, assess_lap, assess_ribbon, clip_at_rest,
                         fitted_centres, fold_height_for_length, polyline_length, reference_length, ribbon_points)


def anchor(side):
    return np.array([-0.006, side * 0.016, 0.052])


def fitted(side=1):
    return ribbon_points(fitted_centres(side, anchor(side)))


@pytest.mark.parametrize("side", [1, -1])
def test_fitted_reference_meets_geometric_assay(side):
    evidence = assess_ribbon(fitted(side), side, anchor(side))
    assert evidence.passes


@pytest.mark.parametrize("side", [1, -1])
def test_loose_material_does_not_pass(side):
    centres = fitted_centres(side, anchor(side))
    floor = reference_length(centres) + SHOULDER_SLACK
    centres[2, 2] += fold_height_for_length(centres, floor)
    e = assess_ribbon(ribbon_points(centres), side, anchor(side), material_floor=floor)
    assert not e.passes
    assert e.pinch["status"] == "fold_witness"
    assert e.pinch["witness"]["material_error_m"] < 1e-7
    assert e.local_overlap_bound_m > 0.002


@pytest.mark.parametrize("side", [1, -1])
def test_flattening_must_not_delete_material(side):
    p = fitted(side)
    floor = polyline_length(p[:, CENTRE_COLUMN]) + SHOULDER_SLACK
    e = assess_ribbon(p, side, anchor(side), material_floor=floor)
    assert not e.material_conserved
    assert not e.passes


@pytest.mark.parametrize("side", [1, -1])
def test_suspended_taut_ribbon_fails_independent_contact_check(side):
    p = fitted(side)
    p[CONTACT_SLICE, :, 2] += 0.025
    e = assess_ribbon(p, side, anchor(side))
    assert not e.against_body
    assert not e.passes


@pytest.mark.parametrize("side", [1, -1])
def test_penetration_is_not_clamped_to_snug(side):
    p = fitted(side)
    p[CONTACT_SLICE, :, 2] -= 0.010
    e = assess_ribbon(p, side, anchor(side))
    assert not e.against_body
    assert e.min_signed_gap_m < -0.002


@pytest.mark.parametrize("side", [1, -1])
def test_half_twist_fails_even_with_no_extra_material(side):
    centres = fitted_centres(side, anchor(side))
    e = assess_ribbon(ribbon_points(centres, math.pi), side, anchor(side))
    assert not e.untwisted
    assert not e.passes


def test_width_edge_can_cross_neck_while_centre_remains_in_corridor():
    p = fitted()
    p[SHOULDER_SLICE, :, 1] -= 0.014
    assert p[CONTROL_ROWS[2], CENTRE_COLUMN, 1] > 0.015
    assert not assess_ribbon(p, 1, anchor(1)).routed


def test_misrouted_shoulder_fails():
    p = fitted()
    p[SHOULDER_SLICE, :, 1] += 0.060
    assert not assess_ribbon(p, 1, anchor(1)).passes


def test_legal_joint_motion_can_create_real_body_penetration():
    c = fitted_centres(1, anchor(1))
    for row, weight in ((1, 1.0), (2, 1.0), (3, 1.0), (4, 0.5), (5, 0.25)):
        c[row, 1] += 0.00475 * weight
    e = assess_ribbon(ribbon_points(c, -0.15), 1, anchor(1))
    assert e.routed
    assert not e.against_body
    assert not e.passes


def test_final_good_state_does_not_mask_later_bad_state():
    p = fitted()
    assert assess_ribbon(p, 1, anchor(1)).passes
    p[CONTACT_SLICE, :, 2] += 0.030
    assert not assess_ribbon(p, 1, anchor(1)).passes


def test_invalid_geometry_fails_closed():
    p = fitted()
    p[3, 1, 0] = np.nan
    assert not assess_ribbon(p, 1, anchor(1)).passes



def test_micrometre_noise_does_not_invalidate_fitted_geometry():
    p = fitted()
    rng = np.random.default_rng(7)
    p += rng.uniform(-1e-6, 1e-6, p.shape)
    assert assess_ribbon(p, 1, anchor(1)).passes


def test_clip_round_trip_motion_during_window_is_not_still():
    history = [0.125] * 21
    history[10] += 0.02
    assert not clip_at_rest(history)


@pytest.mark.parametrize("side", [1, -1])
def test_lap_anchor_cannot_cross_midline(side):
    a = np.array([-0.015, side * 0.085, 0.030])
    b = anchor(side)
    width = np.array([0.0, 0.011, 0.0])
    p = np.array([[a - width, a + width], [b - width, b + width]])
    assert assess_lap(p, side, b)
    p[-1, :, 1] *= -1
    assert not assess_lap(p, side, b)
