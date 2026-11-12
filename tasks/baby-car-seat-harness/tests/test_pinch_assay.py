import math

import numpy as np
import pytest

import harness_fit as hf
import scene_assets as sa
from pinch_assay import (ARC_STEPS, BEND_RADIUS, OVERLAP, PAD_CONTACT_TOLERANCE,
                         assess_patch, lane_bounds, make_witness)


def shoulder(side, feed):
    tongue = sa.to_seat(sa.nominal_tongue(side))[0]
    centres = hf.fitted_centres(side, tongue)
    material = sa.material_budget(side) - feed - hf.lap_length(side, tongue)
    centres[2, 2] += hf.fold_height_for_length(centres, material)
    points = hf.ribbon_points(centres)
    return points[hf.CONTROL_ROWS[1]:hf.CONTROL_ROWS[3] + 1]


def flat_width_path(excess):
    length = .020 + excess
    high = math.sqrt((length / 2) ** 2 - .010 ** 2)
    centre = np.array([[0., 0., 0.], [.010, 0., high], [.020, 0., 0.]])
    patch = np.repeat(centre[:, None], 11, axis=1)
    patch[:, :, 1] += np.linspace(-.011, .011, 11)
    return patch


@pytest.mark.parametrize('side', [1, -1])
def test_nominal_full_width_certificate_does_not_depend_on_reference_route(side):
    result = assess_patch(shoulder(side, .025), hf.BODY)
    assert result.status == 'no_fold_certified'
    assert result.overlap_upper_m < .0009
    assert result.centre_overlap_upper_m == pytest.approx(.00005209775, abs=1e-9)
    assert result.patch_material_m > .0217  # 20 mm is the u projection, not material length


@pytest.mark.parametrize('side', [1, -1])
def test_loose_geometry_has_conserved_finite_pad_witness(side):
    patch = shoulder(side, 0.)
    result = assess_patch(patch, hf.BODY)
    assert result.status == 'fold_witness'
    witness = result.witness
    assert witness.valid and witness.within_body_tolerance
    assert witness.overlap_m >= OVERLAP
    assert witness.layer_spacing_m == pytest.approx(2 * BEND_RADIUS)
    assert witness.material_error_m < 1e-7
    assert witness.endpoint_error_m == 0
    assert witness.opposed_normal_dot <= -.99
    assert witness.pad_contact_error_m <= PAD_CONTACT_TOLERANCE
    np.testing.assert_allclose(witness.points[[0, -1]], patch[[0, -1]], atol=0.)
    np.testing.assert_allclose(np.linalg.norm(np.diff(witness.points, axis=0), axis=2).sum(axis=0),
                               np.linalg.norm(np.diff(patch, axis=0), axis=2).sum(axis=0), atol=1e-12)


@pytest.mark.parametrize('excess, passes', [(.0038, True), (.004, False), (.0042, False)])
def test_strict_two_millimetre_chord_bound(excess, passes):
    result = assess_patch(flat_width_path(excess))
    assert result.overlap_upper_m == pytest.approx(excess / 2 + 1e-9, abs=1e-12)
    assert result.passes is passes


def test_reference_same_length_does_not_hide_a_real_double_fold():
    # A curved reference can have the same length as this 6 mm U. Subtracting
    # reference length would give zero, although the material can form a fold.
    connector = 2 * ARC_STEPS * BEND_RADIUS * math.sin(math.pi / (2 * ARC_STEPS))
    excess = .012 + connector - 2 * BEND_RADIUS
    patch = flat_width_path(excess)
    reference_length = np.linalg.norm(np.diff(patch[:, 5], axis=0), axis=1).sum()
    assert (reference_length - reference_length) / 2 == 0
    result = assess_patch(patch)
    assert result.status == 'fold_witness'
    assert result.witness.overlap_m >= .002
    assert not result.passes


def test_lane_in_triangle_interior_is_not_omitted():
    patch = np.zeros((2, 3, 3))
    patch[:, :, 1] = [-.002, 0., .002]
    patch[1, :, 0] = [.020, -.020, -.020]
    lengths, chords, bounds = lane_bounds(patch)
    assert np.allclose(lengths - chords, 0.)  # all stored column curves are straight
    assert bounds[0] > .0099  # an interior lane has a much shorter endpoint chord
    assert not assess_patch(patch).passes


def test_unproved_witness_is_indeterminate_not_no_fold():
    patch = flat_width_path(.012)
    bodies = (((.010, 0., 0.), (.1, .1, .1)),)
    result = assess_patch(patch, bodies)
    assert result.status == 'indeterminate'
    assert not result.passes
    assert result.witness is not None and not result.witness.within_body_tolerance


def test_invalid_geometry_fails_closed():
    patch = shoulder(1, .025)
    patch[2, 5, 0] = math.nan
    result = assess_patch(patch)
    assert result.status == 'indeterminate' and not result.passes


def test_small_mesh_noise_keeps_the_certificate():
    patch = shoulder(1, .025)
    patch += np.random.default_rng(8).uniform(-1e-6, 1e-6, patch.shape)
    assert assess_patch(patch, hf.BODY).passes


@pytest.mark.parametrize("reverse_length,reverse_width", [(True, False), (False, True), (True, True)])
def test_pad_solids_stay_outside_layer_gap_after_material_order_reversal(reverse_length, reverse_width):
    patch = shoulder(1, 0.)
    if reverse_length:
        patch = patch[::-1]
    if reverse_width:
        patch = patch[:, ::-1]
    witness = make_witness(patch)
    assert witness is not None and witness.valid
    assert witness.pads_disjoint and witness.pads_away_from_gap
    assert witness.pad_separation_lower_m > .0005
