"""The hanging tool may settle about its closing axis while its cut point stays fixed."""
import numpy as np
import pytest
import scene_assets as sa
from supported_pitch import supported_pose,OpenThroat


def test_supported_pitch_preserves_the_material_section_and_does_not_rotate_the_hand():
    R=np.eye(3);P=np.array([.2,.1,.8]);C=P+np.array([.0025,.066,.0018])
    observed=sa._rx(-.1);hand=np.diag([1.,-1.,-1.]);relative=hand.T@observed
    position,rotation,pitch,residual=supported_pose(P,R,C,observed)
    assert np.allclose(position+rotation@np.array([.0025,.066,.0018]),C)
    assert np.allclose(rotation@relative.T,hand)
    assert pitch==pytest.approx(-.1) and residual<1e-7


def test_other_axis_error_is_not_hidden_by_the_free_pitch():
    _,_,_,residual=supported_pose([0,0,0],np.eye(3),[0,.0246,.0018],sa._ry(np.radians(2)))
    assert residual==pytest.approx(np.radians(2))


def test_real_finite_throat_catches_the_boss_and_allows_the_supported_pitch():
    import task
    w=next(w for w in task.demo()['right']if w.label=='squeeze: cut')
    R=w.rotation@np.diag([1.,-1.,-1.]);C=task._cane_point(task.CUT_AT)+[0,0,task.TOP]
    bodyP=np.r_[sa.POT_XY,task.TOP]+sa.BUSH_R@sa.on_cane(sa.SG_BOTTOM_S)
    state={'jaw':0.,'fuse':0.,'plant_body_poses':{'stub_col':{
        'position_m':bodyP.tolist(),'rotation':(sa.BUSH_R@sa._lean()[0]).tolist()}}}
    check=OpenThroat(sa)
    wrong=C-R@np.array([task.ENTRY_CENTRE_X,.015,sa.BLADE_Z])
    assert check(state,wrong,R)['gap_lower_bound_m']<0
    supported=R@sa._rx(-.1)
    proper=C-supported@np.array([task.ENTRY_CENTRE_X,task.CUT_RADIUS,sa.BLADE_Z])
    assert check(state,proper,supported)['gap_lower_bound_m']>0
    # Retain the actual bolt/shoulder surfaces and positive finite clearance
    # across the observed hanging-pitch excursion.
    for angle in (-.09,0.,.025):
        rr=R@sa._rx(angle)
        pp=C-rr@np.array([task.ENTRY_CENTRE_X,task.CUT_RADIUS,sa.BLADE_Z])
        assert check(state,pp,rr)['gap_lower_bound_m']>.0005
    with pytest.raises(ValueError,match='Missing actual material'):
        check({},proper,supported)


def test_final_seating_change_does_not_move_the_known_clear_opening_or_mouth(monkeypatch):
    import task
    before={w.label:w for w in task.demo()['right']}
    monkeypatch.setattr(task,'ENTRY_CENTRE_X',task.ENTRY_CENTRE_X+.001)
    after={w.label:w for w in task.demo()['right']}
    for label in ('turn the closed secateurs','recenter vertically','open beside the cane',
                  'beside the held cane','follow the open hook throat'):
        assert np.allclose(before[label].pos,after[label].pos,atol=1e-12,rtol=0)
        assert np.allclose(before[label].rotation,after[label].rotation,atol=1e-12,rtol=0)
    assert np.linalg.norm(np.array(before['squeeze: cut'].pos)-after['squeeze: cut'].pos)==pytest.approx(.001)
