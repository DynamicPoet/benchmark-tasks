"""Ordinary orientation actions and finite steel preserve the task's physical setup."""
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from pose_motion import PoseWaypoint,PoseArmPlan,replace_task_plans
from yam_arena.demo import ArmPlan
from yam_arena import station
import scene_assets as s


def test_task_local_pose_adapter_does_not_replace_foreign_plans():
    w=PoseWaypoint(.5,(.3,0,.9),0.,rotation=np.eye(3))
    p=ArmPlan([w],station.BASES['right'])
    policy=SimpleNamespace(_task=SimpleNamespace(path=Path('/other/task.py')),_plans={'right':p},_bases=station.BASES)
    replace_task_plans(policy,Path('/this/task.py'))
    assert policy._plans['right'] is p
    policy._task.path=Path('/this/task.py');replace_task_plans(policy,Path('/this/task.py'))
    assert isinstance(policy._plans['right'],PoseArmPlan)


def test_closed_transit_uses_continuous_bounded_joint_actions_and_matching_fk():
    import task
    from yam_arena import ik
    waypoints=task.demo()['right']
    w=next(w for w in waypoints if w.label=='turn the closed secateurs')
    plan=PoseArmPlan([w],station.BASES['right'])
    for t in np.linspace(0,w.duration,31)[:-1]:
        action,_=plan.action7(float(t))
        expected=w.joint_start+ik.smoothstep(t/w.duration)*(w.joint_goal-w.joint_start)
        assert np.allclose(action[:6],expected)
        assert np.all(action[:6]>=ik.LOWER) and np.all(action[:6]<=ik.UPPER)
        p,r,_,_=plan.target(t)
        actual_p,actual_r=ik.tcp_pose(action[:6],station.BASES['right'])
        assert np.allclose(p,actual_p) and np.allclose(r,actual_r)


def test_carry_is_closed_and_entry_is_actual_tool_plane_reference():
    import task
    waypoints=task.demo()['right'];bylabel={w.label:w for w in waypoints}
    for name in ('close for carrying','lift closed','turn the closed secateurs','back out closed','turn back closed'):
        assert bylabel[name].finger==task.FINGER_CLOSED
    assert bylabel['recenter vertically'].finger==-.021
    assert bylabel['beside the held cane'].finger==task.FINGER_OPEN
    assert bylabel['open beside the cane'].finger<task.FINGER_CLOSED
    w=bylabel['squeeze: cut'];relative=np.diag([1.,-1.,-1.]);R=w.rotation@relative
    cane=R.T@s.BUSH_R@s.CANE_DIR
    assert abs(cane[0])<1e-4 and R[2,1]<-.96
    point=np.array(w.pos)+R@np.array([task.ENTRY_CENTRE_X,task.GRIP_BACK+task.CUT_RADIUS,s.BLADE_Z-(task.TCP_OVER_TOOL+task.TOOL_SAG)])
    expected=task._cane_point(task.CUT_AT)+[0.,0.,task.TOP]
    assert np.linalg.norm(point-expected)<1e-12


def test_counterblade_keeps_actual_steel_but_not_the_filled_throat():
    hook=next(m for n,m,_ in s._closed_half(-1)[0] if n=='hook')
    pieces=s.convex_hook_sections(hook)
    assert len(pieces)==4
    equations=[ConvexHull(m.v).equations for m in pieces]
    samples=np.vstack([hook.v,hook.v[hook.f].mean(axis=1)])
    for p in samples:
        assert any(np.max(eq[:,:3]@p+eq[:,3])<2e-8 for eq in equations)
    # This is empty throat in the actual visible counterblade, filled by one big hull.
    throat=np.array([.0012,.020,-.0032]);old=ConvexHull(hook.v).equations
    assert np.max(old[:,:3]@throat+old[:,3])<0
    assert all(np.max(eq[:,:3]@throat+eq[:,3])>0 for eq in equations)


def test_extended_stem_sleeves_are_hidden_and_variants_keep_their_meaning(monkeypatch):
    import task
    assert s.T_LOW_S+s.CG_HIDE+s.L_CG < s.TOP_S
    assert s.T_LOW_S+s.CB_HIDE+s.L_CB < s.TOP_S
    offsets={'success':('squeeze: cut',s.BUD_TOP_S+.009),
             'brown':('squeeze: cut',s.BROWN_S+.011),
             'repair':('second cut',s.BUD_TOP_S+.006),
             'retrim':('second cut',s.BUD_TOP_S+.003),
             'recut':('second cut',s.BUD_S-.005)}
    for mode,(label,expected) in offsets.items():
        monkeypatch.setenv('TASK_DEMO',mode)
        w=next(w for w in task.demo()['right']if w.label==label)
        R=w.rotation@np.diag([1.,-1.,-1.])
        point=np.array(w.pos)+R@np.array([task.ENTRY_CENTRE_X,task.GRIP_BACK+task.CUT_RADIUS,s.BLADE_Z-(task.TCP_OVER_TOOL+task.TOOL_SAG)])
        origin=np.array([*s.CANE_BASE,task.TOP+s.SOIL_Z])
        assert abs((point-origin)@(s.BUSH_R@s.CANE_DIR)-expected)<1e-12


def test_live_cane_counterexample_keeps_valid_full_pose_waypoints(monkeypatch):
    import task
    monkeypatch.setenv('TASK_DEMO','green')
    plans=task.demo()
    assert all(isinstance(w,PoseWaypoint) for side in plans.values() for w in side)
    left={w.label:w for w in plans['left']}
    assert np.array_equal(left['hold the cane'].rotation,left['hold it'].rotation)
    assert left['back off past the tip'].joint_goal is not None


def test_closed_hand_withdrawal_preserves_the_real_wrist_then_joins_the_known_free_return():
    from pose_motion import closed_hand_pose
    from yam_arena import ik
    nominal=np.array([.2,.03,.9]);R0=s._rx(.12)
    capture={'TCP':[.201,.035,.91],'R':R0.tolist(),'nominal_TCP':nominal.tolist()}
    boundaries={('right','to a free spot'):(1.,2.)}
    p,R=closed_hand_pose(capture,'back out closed',0.,boundaries,nominal,np.eye(3))
    assert np.allclose(p,capture['TCP']) and np.allclose(R,R0)
    free=np.array([.15,-.02,1.])
    p,R=closed_hand_pose(capture,'to a free spot',2.,boundaries,free,np.eye(3))
    assert np.allclose(p,free) and np.allclose(R,np.eye(3))


def test_drop_counterexample_preserves_normal_bilateral_cutting_prefix(monkeypatch):
    import task
    monkeypatch.setenv('TASK_DEMO','success');normal=task.demo()
    monkeypatch.setenv('TASK_DEMO','failure');drop=task.demo()
    for side,stop in (('right','squeeze: cut'),('left','hold the cutting')):
        for a,b in zip(normal[side],drop[side]):
            assert a.label==b.label and a.duration==b.duration and a.finger==b.finger
            assert np.allclose(a.pos,b.pos,atol=1e-12,rtol=0)
            assert np.allclose(a.rotation,b.rotation,atol=1e-12,rtol=0)
            if a.label==stop:break
    release=next(w for w in drop['left']if w.label=='release outside the bin')
    hold=next(w for w in drop['left']if w.label=='hold the cutting')
    assert np.allclose(release.pos,hold.pos) and release.finger==task.FINGER_OPEN
    assert np.linalg.norm(np.array(release.pos)[:2]-task.sa.BIN_XY)>.13
