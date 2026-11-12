import numpy as np
import pytest

import harness_fit as hf
import scene_assets as sa


@pytest.mark.parametrize('side', [1, -1])
def test_buckling_without_tail_pull_cannot_consume_material(side):
    budget = sa.material_budget(side)
    totals = []
    excess = []
    for world_tongue in (sa.tongue_start(side), sa.nominal_tongue(side)):
        tongue = sa.to_seat(world_tongue)[0]
        centres = hf.fitted_centres(side, tongue)
        floor = budget - hf.lap_length(side, tongue)
        centres[2, 2] += hf.fold_height_for_length(centres, floor)
        points = hf.ribbon_points(centres)
        actual = hf.polyline_length(points[:, hf.CENTRE_COLUMN])
        totals.append(actual + hf.lap_length(side, tongue))
        excess.append(actual - hf.reference_length(hf.fitted_centres(side, tongue)))
    np.testing.assert_allclose(totals, [budget, budget], atol=1e-9)
    assert excess[1] > excess[0]
    assert excess[1] == pytest.approx(hf.SHOULDER_SLACK, abs=1e-9)


@pytest.mark.parametrize('side', [1, -1])
def test_only_actual_material_feed_reduces_fixed_budget(side):
    tongue = sa.to_seat(sa.nominal_tongue(side))[0]
    ref = hf.fitted_centres(side, tongue)
    lap = hf.lap_length(side, tongue)
    for actual_feed in (0.0, hf.SHOULDER_SLACK / 2, hf.SHOULDER_SLACK):
        c = ref.copy()
        floor = sa.material_budget(side) - actual_feed - lap
        c[2, 2] += hf.fold_height_for_length(c, floor)
        points = hf.ribbon_points(c)
        assert hf.polyline_length(points[:, hf.CENTRE_COLUMN]) + lap == pytest.approx(sa.material_budget(side) - actual_feed, abs=1e-9)
