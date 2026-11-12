import math

import pytest

from pruning_checks import PruningState

BUD = 0.059
GOOD = BUD + 0.009


def test_good_first_cut_requires_good_final_stub_and_original_bud():
    state = PruningState(BUD)
    state.cut(GOOD)
    assert state.credit(GOOD, True) == 1.0
    assert state.credit(GOOD, False) == 0.0
    assert state.credit(BUD + 0.002, True) == 0.25
    assert state.credit(BUD - 0.002, True) == 0.0


@pytest.mark.parametrize("height", [BUD, BUD - 0.001, 0.045])
def test_later_cut_removing_bud_fails_permanently(height):
    state = PruningState(BUD)
    state.cut(GOOD)
    state.cut(height)
    assert state.bud_damaged
    assert state.credit(GOOD, True) == 0.0
    state.cut(GOOD)
    assert state.credit(GOOD, True) == 0.0


def test_first_cut_removing_bud_also_fails():
    state = PruningState(BUD)
    state.cut(BUD - 0.004)
    assert state.credit(BUD - 0.004, False) == 0.0


def test_bad_first_cut_cannot_be_repaired_into_full_credit():
    state = PruningState(BUD)
    state.cut(0.099)
    state.cut(GOOD)
    assert state.first_cut == 0.099
    assert state.stub_length == GOOD
    assert state.credit(GOOD, True) == 0.25


def test_harmless_retrim_stays_eligible():
    state = PruningState(BUD)
    state.cut(GOOD)
    state.cut(BUD + 0.006)
    assert state.credit(BUD + 0.006, True) == 1.0


def test_too_short_retrim_loses_full_credit_without_inventing_bud_loss():
    state = PruningState(BUD)
    state.cut(GOOD)
    state.cut(BUD + 0.002)
    assert not state.bud_damaged
    assert state.credit(BUD + 0.002, True) == 0.25


def test_displaced_bud_put_back_is_still_failed():
    state = PruningState(BUD)
    state.cut(GOOD)
    state.observe_bud(False)
    state.observe_bud(True)
    assert state.credit(GOOD, True) == 0.0


@pytest.mark.parametrize("height", [math.nan, math.inf, -math.inf])
def test_non_finite_cut_fails(height):
    state = PruningState(BUD)
    state.cut(GOOD)
    state.cut(height)
    assert state.credit(GOOD, True) == 0.0


def test_no_cut_and_reset_state():
    state = PruningState(BUD)
    assert state.credit(0.24, True) == 0.0
    state.cut(BUD - 0.001)
    fresh = PruningState(BUD)
    fresh.cut(GOOD)
    assert fresh.credit(GOOD, True) == 1.0
