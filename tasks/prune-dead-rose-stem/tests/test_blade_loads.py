"""A loaded finite target is necessary; a blocked tool elsewhere is insufficient."""
import numpy as np
import pytest
from blade_contact import target_load


def row(force=150., body='stub_col', point=(0., .018, 0.), separation=0.):
    return {'body':body,'point_world_m':point,'force_on_blade_N':[force,0.,0.],
            'signed_normal_N':-force,'separation_m':separation}


def load(rows):
    return target_load(rows,[0.,.018,-.02],[0.,.018,.02],.00225,[0.,0.,0.],[0.,0.,1.],{'stub_col'})


def test_measured_normal_load_uses_actual_closing_lever_not_radius_guess():
    a=load([row()]);assert a['normal_load_N']==150.
    assert a['resisting_moment_Nm']==pytest.approx(2.7)
    assert a['effective_lever_m']==pytest.approx(.018)


@pytest.mark.parametrize('contact',[
    row(body='table'),row(point=(.04,.018,0.)),row(separation=.001),row(force=-150.)])
def test_wrong_body_far_point_speculative_gap_or_opening_force_cannot_cut(contact):
    assert load([contact])['normal_load_N']==0.


@pytest.mark.parametrize('fraction,cut',[(0.,False),(.99,False),(1.1,True)])
def test_cut_needs_current_measured_target_load_as_well_as_loaded_gauge(monkeypatch,fraction,cut):
    import task,load_fixture
    from test_cut_contact import cutting_setup
    task._clear();original=load_fixture.measured_contacts
    def scaled(env):
        rows=original(env)
        for r in rows:r['force_on_blade_N']=(np.asarray(r['force_on_blade_N'])*fraction/2).tolist()
        return rows
    monkeypatch.setattr(load_fixture,'measured_contacts',scaled)
    observe,_=cutting_setup(monkeypatch)
    observe(0.);observe(.24)
    assert not task._S['cut']  # One impact/sample cannot sever it.
    observe(.24)
    assert task._S['cut']==cut


def test_gauge_loaded_by_another_body_never_substitutes_for_target_contact(monkeypatch):
    import task,load_fixture
    from test_cut_contact import cutting_setup
    task._clear();original=load_fixture.measured_contacts
    def wrong(env):
        rows=original(env)
        for r in rows:r['body']='table'
        return rows
    monkeypatch.setattr(load_fixture,'measured_contacts',wrong)
    observe,_=cutting_setup(monkeypatch)
    for q in (0.,.24,.24):observe(q)
    assert not task._S['cut'] and not task._S['cut_events']


def test_blunt_blade_spine_load_outside_current_aperture_cannot_cut(monkeypatch):
    import task
    from test_cut_contact import cutting_setup
    task._clear();observe,_=cutting_setup(monkeypatch)
    # Material is against the solid back of the blade, still within radial reach.
    # The fake measured load and the gauge are high; the current jaw side is wrong.
    for q in (0.,.24,.24):observe(q,offset=(-.015,0.,0.))
    assert not task._S['cut'] and task.score({})==0.


def test_planned_open_entry_and_material_engagement_budget():
    import math,task
    assert task.sa.SPRING_PRELOAD==pytest.approx(.30)
    opening=task.sa.JAW_OPEN-task.CARRY_PRECLOSE
    assert 2*task.CUT_RADIUS*math.sin(opening/2)>2*task.sa.R0+2*.0002
    assert task._handle_outer_x(-(task.GRIP_BACK+.018),0.) <= abs(task.FINGER_OPEN)
    maximum_spring=task.sa.SPRING_K*(task.sa.CLOSE+task.sa.SPRING_PRELOAD)
    # This nominal simultaneous upper-bound budget is not a certificate of
    # actual pad contact lines. Native caps and measured target load still rule.
    assert (task.F_CUT*task.CUT_RADIUS+maximum_spring)/(task.GRIP_BACK+.018)<40.
    assert task.CUT_RADIUS==pytest.approx(.0246)
