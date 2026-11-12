import math

import numpy as np
import pytest

import task
from harness_fit import assess_ribbon, fitted_centres, ribbon_points


def evidence(side, lift=0.0, twist=0.0):
    tongue = np.array([-0.006, side * 0.016, 0.052])
    centres = fitted_centres(side, tongue)
    centres[1:6, 2] += lift
    value = assess_ribbon(ribbon_points(centres, twist), side, tongue).record()
    value['lap_routed'] = True
    return value


@pytest.fixture
def complete_state():
    task._clear()
    task._S.update(head=False, seat=False, latched={'l': True, 'r': True}, closed=True,
                   clip_u=task.sa.ARMPIT_U, tip_to_seat=0.1, tip_to_adjuster=0.1,
                   history=[task.sa.ARMPIT_U] * 21,
                   full_history=[True] * 21,
                   fit={'l': evidence(1), 'r': evidence(-1)})
    return task._S


def test_both_shoulders_required_for_full_credit(complete_state):
    assert task.score({}) == 1.0
    complete_state['fit']['l'] = evidence(1, lift=0.025)
    assert task.score({}) == 0.75
    complete_state['fit']['l'] = evidence(1)
    complete_state['fit']['r'] = evidence(-1, twist=math.pi)
    assert task.score({}) == 0.75


def test_hardware_progress_is_retained(complete_state):
    complete_state['latched']['r'] = False
    assert task.score({}) == 0.25
    complete_state['latched']['r'] = True
    complete_state['closed'] = False
    assert task.score({}) == 0.5


def test_holding_the_adjustment_tail_is_not_completed_fit(complete_state):
    complete_state['tip_to_adjuster'] = 0.0
    assert task.score({}) == 0.75


def test_forbidden_contact_overrides_otherwise_good_fit(complete_state):
    complete_state['head'] = True
    assert task.score({}) == 0.0


def test_prior_good_fit_does_not_preserve_full_credit(complete_state):
    assert task.score({}) == 1.0
    complete_state['fit']['r'] = evidence(-1, lift=0.030)
    assert task.score({}) == 0.75


def test_clip_out_and_back_is_not_still(complete_state):
    complete_state['history'][10] += 0.020
    assert task.score({}) == 0.75


def test_pulling_before_buckling_cannot_hide_material_overdraw(complete_state, monkeypatch):
    from types import SimpleNamespace

    limits = []
    asset = SimpleNamespace(device="cpu", joint_names=["adjuster_pull"],
                            write_joint_position_limit_to_sim_index=lambda **kw: limits.append(kw["limits"].numpy()))
    monkeypatch.setattr(task, "_drive", lambda *a, **kw: None)
    open_tongues = {tag: task.sa.to_seat(task.sa.tongue_start(side))[0] for tag, side in task.TAGS.items()}
    closed_tongues = {tag: task.sa.to_seat(task.sa.nominal_tongue(side))[0] for tag, side in task.TAGS.items()}
    # A deliberately injected measured overdraw bypasses the physical stop.
    # It must remain disqualifying after both tongues reach the buckle.
    task._constrain_material_feed(asset, open_tongues, 0.025)
    assert limits[-1][0, 0, 1] < 0.010
    assert complete_state["material_overdraw"]
    task._constrain_material_feed(asset, closed_tongues, 0.025)
    assert limits[-1][0, 0, 1] == pytest.approx(0.025)
    assert all(e["passes"] for e in complete_state["fit"].values())
    assert task.score({}) == 0.75


def test_normal_feed_limit_does_not_use_fold_tracking_error(complete_state, monkeypatch):
    from types import SimpleNamespace

    asset = SimpleNamespace(device="cpu", joint_names=["adjuster_pull"], write_joint_position_limit_to_sim_index=lambda **kw: None)
    monkeypatch.setattr(task, "_drive", lambda *a, **kw: None)
    tongues = {tag: task.sa.to_seat(task.sa.nominal_tongue(side))[0] for tag, side in task.TAGS.items()}
    task._constrain_material_feed(asset, tongues, 0.025)
    assert not complete_state["material_overdraw"]
    assert task.score({}) == 1.0


def test_only_last_frame_meeting_all_conditions_is_not_complete(complete_state):
    complete_state['full_history'] = [False] * 20
    task._sample_completion()
    assert task.score({}) == 0.75
    for _ in range(20):
        task._sample_completion()
    assert task.score({}) == 1.0


@pytest.mark.parametrize('fault', ['shoulder', 'tail_grasp'])
def test_one_bad_frame_requires_a_new_complete_second(complete_state, fault):
    if fault == 'shoulder':
        complete_state['fit']['r'] = evidence(-1, lift=0.025)
    else:
        complete_state['tip_to_adjuster'] = 0.0
    task._sample_completion()
    complete_state['fit']['r'] = evidence(-1)
    complete_state['tip_to_adjuster'] = 0.1
    for _ in range(20):
        task._sample_completion()
    assert task.score({}) == 0.75
    task._sample_completion()
    assert task.score({}) == 1.0


def test_repeated_score_queries_cannot_satisfy_the_hold(complete_state):
    complete_state['full_history'] = [False] * 20
    task._sample_completion()
    before = list(complete_state['full_history'])
    for _ in range(50):
        assert task.score({}) == 0.75
    assert complete_state['full_history'] == before
