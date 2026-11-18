from types import SimpleNamespace
import numpy as np
import pytest

from pruning_control import bounded_position,PruningController
from pose_motion import PoseArmPlan
from yam_arena import station,ik
from yam_arena.constants import READY


def test_reference_integral_removes_a_constant_position_load_without_gain_changes():
    desired=np.full(6,.2);actual=previous=np.zeros(6);bias=np.zeros(6)
    for _ in range(800):
        sent,bias=bounded_position(desired,actual,np.zeros(6),np.zeros(6),
            np.full(6,10.),np.full(6,1.5),np.full(6,7.),
            np.tile([-1.,1.],(6,1)),np.full(6,5.),previous,bias,.05)
        # A simple stationary physical spring under a constant opposing load.
        actual=sent-.06;previous=sent
    assert np.max(np.abs(actual-desired))<1e-6
    assert np.allclose(bias,.06,atol=1e-6)


def test_final_reference_stays_in_all_intervals_and_windup_does_not_continue():
    sent,bias=bounded_position(np.ones(6),np.zeros(6),np.zeros(6),np.zeros(6),
        np.full(6,10.),np.ones(6),np.full(6,.5),np.tile([-.1,.1],(6,1)),
        np.full(6,.2),np.zeros(6),np.zeros(6),.05)
    assert np.max(sent)<=.01+1e-12
    assert np.max(np.abs(10*sent))<=.5
    assert np.all(bias==0)
    with pytest.raises(ValueError,match='intersection'):
        bounded_position(np.ones(6),np.ones(6),np.zeros(6),np.zeros(6),
            np.full(6,10.),np.ones(6),np.full(6,.1),np.tile([-1.,1.],(6,1)),
            np.full(6,.1),np.zeros(6),np.zeros(6),.05)


def setup_controller():
    import task
    plans={s:PoseArmPlan(w,station.BASES[s])for s,w in task.demo().items()}
    policy=SimpleNamespace(_plans=plans,_end=max(p.duration for p in plans.values()),hold_s=3.)
    state={'robots':{},'tool_o':[.25,0.,task.TOP+.15],'tool_R':np.eye(3).tolist(),
           'both_handles':True,'both_target_fingers':True,'blocking_left':[],'severed':False,'jaw':.51,
           'handle_pad_contact_span_m':.035}
    for side,p in plans.items():
        state['robots'][side]={'q':np.array(READY).tolist(),'qd':[0.]*6,
            'TCP':p.start_pos.tolist(),'R':p.start_rot.tolist(),'fingers':[-.0175,-.0175],
            'gravity_Nm':[0.]*6,'K':[10.]*6,'D':[1.5]*6,'effort':[27.]*3+[7.]*3,
            'finger_limits':[[-.04695,0.],[-.04695,0.]],
            'limits':np.stack([ik.LOWER,ik.UPPER],axis=1).tolist(),
            'speed':[5.5]*3+[20.]*3}
    detail={}
    ctl=PruningController(policy,lambda:state,lambda d:detail.update(d),task.sa,
        {'grip_back':task.GRIP_BACK,
         'tcp_offset':task.TCP_OVER_TOOL+task.TOOL_SAG,'table_top':task.TOP,
         'cut_section':[task.ENTRY_CENTRE_X,task.CUT_RADIUS,task.sa.BLADE_Z]})
    ctl.open_throat=lambda *args:{'gap_lower_bound_m':.001,'part':'interface fixture'}
    return ctl,state,detail


def observation(t=.05):return SimpleNamespace(state_time=t)


def test_left_arrival_gate_keeps_fingers_open_instead_of_closing_on_a_miss():
    ctl,state,_=setup_controller();start=ctl.boundaries[('left','hold it')][0]
    ctl.clock=start-.01
    action=ctl.act(observation()).actions[0]
    assert ctl.clock==pytest.approx(start)
    assert action.data[6]>.5
    assert ctl.gate_name==('left','hold it')


def test_recenter_does_not_relax_before_actual_bilateral_stem_grasp():
    ctl,state,_=setup_controller();key=('right','recenter vertically')
    ctl.clock=ctl.boundaries[key][0]-.01;state['both_target_fingers']=False
    action=ctl.act(observation()).actions[0]
    assert ctl.clock==pytest.approx(ctl.boundaries[key][0])
    assert action.data[13]==0.


@pytest.mark.parametrize('back,cross',[(.1013,0.),(.09,.0145)])
def test_nearclosed_pose_is_not_a_substitute_for_observed_full_opening(back,cross):
    ctl,state,_=setup_controller();key=('right','beside the held cane')
    ctl.clock=ctl.boundaries[key][0]-.01
    p,R,_,_=ctl.policy._plans['right'].target(ctl.boundaries[key][0])
    tool_R=R@np.diag([1.,-1.,-1.]);tool_P=p-tool_R@np.array([cross,-back,.0046])
    state['robots']['right'].update(TCP=p.tolist(),R=R.tolist())
    state.update(tool_o=tool_P.tolist(),tool_R=tool_R.tolist())
    ctl.act(observation())
    assert ctl.relative is None
    assert ctl.clock==pytest.approx(ctl.boundaries[key][0])


def test_open_stop_without_real_handle_contact_cannot_unlock_insertion():
    ctl,state,_=setup_controller();key=('right','beside the held cane')
    ctl.clock=ctl.boundaries[key][0]-.01
    ctl.arrived.add(('right','open beside the cane'))
    state.update(jaw=0.,both_handles=False)
    ctl.act(observation())
    assert ctl.clock==pytest.approx(ctl.boundaries[key][0])
    assert key not in ctl.arrived


def test_unexpected_contact_stops_without_powering_a_reference_through_it():
    ctl,state,detail=setup_controller();state['blocking_left']=[{'other':'stem_tip'}]
    action=ctl.act(observation()).actions[0]
    assert action.meta['request_stop']
    assert np.all(ctl.bias['left']==0.)
    assert detail['error']


def test_hook_mouth_waypoint_keeps_observed_open_grip_and_actual_relative_placement(monkeypatch):
    ctl,state,detail=setup_controller()
    start,end=ctl.boundaries[('right','follow the open hook throat')]
    ctl.clock=(start+end)/2
    ctl.relative=(np.array([.003,-.092,.004]),np.diag([1.,-1.,-1.]))
    ctl.capture={'hand_to_tool_p_m':ctl.relative[0].tolist(),'hand_to_tool_R':ctl.relative[1].tolist()}
    ctl.open_reference=-.0449
    ctl.arrived.update({('right','open beside the cane'),('right','beside the held cane')})
    p,R,_,_=ctl.policy._plans['right'].target(ctl.clock)
    tool_R=R@np.diag([1.,-1.,-1.])
    tool_P=p-tool_R@np.array([0.,-ctl.constants['grip_back'],ctl.constants['tcp_offset']])
    hand_R=tool_R@ctl.relative[1].T;hand_P=tool_P-hand_R@ctl.relative[0]
    state.update(tool_o=tool_P.tolist(),tool_R=tool_R.tolist())
    state['robots']['right'].update(TCP=hand_P.tolist(),R=hand_R.tolist())
    captured=[]
    def solve(p,q,seed,**kwargs):
        captured.append((np.array(p),np.array(q)))
        return np.array(seed),0.,0.
    monkeypatch.setattr(ik,'solve',solve)
    action=ctl.act(observation()).actions[0]
    p,R,_,label=ctl.policy._plans['right'].target(ctl.clock)
    assert label=='follow the open hook throat'
    assert action.data[13]==pytest.approx(ctl.open_reference/__import__('yam_arena.constants',fromlist=['FINGER_OPEN']).FINGER_OPEN)
    tool_R=R@np.diag([1.,-1.,-1.])
    offset=np.array([0.,-ctl.constants['grip_back'],ctl.constants['tcp_offset']])
    desired_R=tool_R@ctl.relative[1].T
    desired_p=p-tool_R@offset-desired_R@ctl.relative[0]
    assert np.allclose(captured[-1][0],desired_p)
    assert abs(captured[-1][1]@ik.matrix_to_quat_xyzw(desired_R))==pytest.approx(1.)


def test_live_observed_creep_pauses_the_phase_and_corrects_the_tool_not_an_imaginary_attachment(monkeypatch):
    ctl,state,detail=setup_controller()
    start,end=ctl.boundaries[('right','follow the open hook throat')];ctl.clock=(start+end)/2
    ctl.relative=(np.array([.003,-.092,.004]),np.diag([1.,-1.,-1.]))
    ctl.capture={'hand_to_tool_p_m':ctl.relative[0].tolist(),'hand_to_tool_R':ctl.relative[1].tolist()}
    ctl.open_reference=-.0449
    ctl.arrived.update({('right','open beside the cane'),('right','beside the held cane')})
    p,R,_,_=ctl.policy._plans['right'].target(ctl.clock);tool_R=R@np.diag([1.,-1.,-1.])
    tool_P=p-tool_R@np.array([0.,-ctl.constants['grip_back'],ctl.constants['tcp_offset']])
    hand_R=tool_R@ctl.relative[1].T;hand_P=tool_P-hand_R@ctl.relative[0]
    state['robots']['right'].update(TCP=hand_P.tolist(),R=hand_R.tolist())
    a=np.radians(1.);creep=np.array([[np.cos(a),-np.sin(a),0.],[np.sin(a),np.cos(a),0.],[0.,0.,1.]])
    actual_tool_R=tool_R@creep;actual_tool_P=tool_P+.0007*tool_R[:,0]
    state.update(tool_o=actual_tool_P.tolist(),tool_R=actual_tool_R.tolist())
    captured=[]
    monkeypatch.setattr(ik,'solve',lambda p,q,seed,**kw:(captured.append((np.array(p),np.array(q)))or(np.array(seed),0.,0.)))
    clock=ctl.clock;ctl.act(observation())
    measured_p=hand_R.T@(actual_tool_P-hand_P);measured_R=hand_R.T@actual_tool_R
    goalR=tool_R@measured_R.T;goalP=tool_P-goalR@measured_p
    assert ctl.clock==clock and detail['actual_tool_placement']['phase_held']
    assert np.allclose(captured[-1][0],goalP)
    assert abs(captured[-1][1]@ik.matrix_to_quat_xyzw(goalR))==pytest.approx(1.)
    assert np.allclose(state['tool_o'],actual_tool_P) and ctl.error is None
    __import__('json').dumps(detail)  # the real atomic diagnostic must accept the NumPy-derived predicate
    first=captured[-1][0].copy();ctl.act(observation(.10))
    assert np.allclose(captured[-1][0],first)  # no accumulated stale-pose correction


def test_wrong_plant_contact_stops_with_measured_robot_positions():
    ctl,state,detail=setup_controller()
    state['wrong_tool_plant_contacts']=[{'sensor':'hook_half','other':'pot','force':[.2,0.,0.]}]
    action=ctl.act(observation()).actions[0]
    assert action.meta['request_stop']
    assert detail['error']=='Tool contacted a non-target plant surface'
    assert np.allclose(action.data[7:13],state['robots']['right']['q'])


@pytest.mark.parametrize('label,allowed',[('hold',True),('follow the open hook throat',False)])
def test_legitimate_closing_shift_is_not_confused_with_the_strict_open_entry_envelope(label,allowed):
    ctl,state,detail=setup_controller()
    start,end=ctl.boundaries[('right',label)];ctl.clock=(start+end)/2
    Rrel=np.diag([1.,-1.,-1.]);opened=np.array([.003,-.09177,.004])
    ctl.capture={'hand_to_tool_p_m':opened.tolist(),'hand_to_tool_R':Rrel.tolist()}
    ctl.relative=(np.array([.003,-.099,.004]),Rrel)
    current=np.array([.003,-.0995,.004])
    p,R,_,_=ctl.policy._plans['right'].target(ctl.clock);tool_R=R@Rrel
    tool_P=p-tool_R@np.array([0.,-ctl.constants['grip_back'],ctl.constants['tcp_offset']])
    hand_R=tool_R@Rrel.T;hand_P=tool_P-hand_R@current
    state['robots']['right'].update(TCP=hand_P.tolist(),R=hand_R.tolist())
    state.update(tool_o=tool_P.tolist(),tool_R=tool_R.tolist(),jaw=.51 if allowed else 0.)
    action=ctl.act(observation()).actions[0]
    if allowed:
        assert ctl.error is None
        assert np.allclose(ctl.relative[0],current)
        assert detail['actual_tool_placement']['grasp_shift_m']==pytest.approx(.0005)
    else:
        assert action.meta['request_stop']
        assert ctl.error=='Working grasp moved outside the bounded placement reference'


@pytest.mark.parametrize('phase,sensor,other,allowed',[
    ('round the handles','tip_left','table',True),
    ('lift closed','tip_left','table',True),
    ('rise','tip_left','table',True),
    ('jaws round it, above the bud','tip_left','table',False),
    ('round the handles','gripper','table',False),
    ('round the handles','tip_left','pot',False)])
def test_real_table_tip_support_is_only_allowed_in_the_table_handling_phase(phase,sensor,other,allowed):
    ctl,state,detail=setup_controller();start,end=ctl.boundaries[('right',phase)];ctl.clock=(start+end)/2
    table='/World/envs/env_0/table/geometry/mesh'
    row={'sensor':f'/World/envs/env_0/RightArm/{sensor}','other':table if other=='table'else'/bush/pot',
         'point':[.1211986095,-.1275345683,.747500062],
         'force':[.000633263,-.006045464,4.750093937],'separation':.000000241704}
    state['blocking_right']=[row];state['robot_contact_detail']={'table_paths':[table]}
    action=ctl.act(observation()).actions[0]
    if allowed:
        assert ctl.error is None and detail['expected_table_tip_support']==[row]
        assert not detail['blocking_right_contacts']
    else:
        assert action.meta['request_stop'] and detail['blocking_right_contacts']==[row]


def test_continuous_rolling_with_fixed_tool_frame_grip_is_not_false_axial_slip():
    ctl,state,detail=setup_controller();start,end=ctl.boundaries[('right','follow the open hook throat')]
    ctl.clock=(start+end)/2;R0=np.diag([1.,-1.,-1.]);p0=np.array([.003,-.09177,.004]);location=-R0.T@p0
    def rz(a):
        a=np.radians(a);return np.array([[np.cos(a),-np.sin(a),0.],[np.sin(a),np.cos(a),0.],[0.,0.,1.]])
    previous=R0@rz(3.);current=R0@rz(3.2)
    ctl.relative=(-previous@location,previous)
    ctl.capture={'hand_to_tool_p_m':p0.tolist(),'hand_to_tool_R':R0.tolist()}
    p,R,_,_=ctl.policy._plans['right'].target(ctl.clock);tool_R=R@R0
    tool_P=p-tool_R@np.array([0.,-ctl.constants['grip_back'],ctl.constants['tcp_offset']])
    hand_R=tool_R@current.T;hand_P=tool_P+tool_R@location
    state.update(tool_o=tool_P.tolist(),tool_R=tool_R.tolist())
    state['robots']['right'].update(TCP=hand_P.tolist(),R=hand_R.tolist())
    ctl.act(observation())
    assert ctl.error is None and detail['actual_tool_placement']['grasp_shift_m']<1e-12
    assert detail['actual_tool_placement']['grasp_rotation_rad']==pytest.approx(np.radians(.2))
    assert detail['actual_tool_placement']['rotation_from_open_capture_rad']==pytest.approx(np.radians(3.2))


def test_feedback_free_travel_can_progress_but_the_R84_boundary_requires_precise_arrival():
    ctl,state,detail=setup_controller();R0=np.diag([1.,-1.,-1.]);relative_p=np.array([.003,-.092,.004])
    ctl.relative=(relative_p,R0);ctl.capture={'hand_to_tool_p_m':relative_p.tolist(),'hand_to_tool_R':R0.tolist()}
    state['jaw']=0.;state['robots']['right']['fingers']=[-.0446,-.0453]
    state['handle_pad_contact_span_m']=.0899
    start,end=ctl.boundaries[('right','beside the held cane')]
    def observed_at(t,offset):
        p,R,_,_=ctl.policy._plans['right'].target(t);tool_R=R@R0
        tool_P=p-tool_R@np.array([0.,-ctl.constants['grip_back'],ctl.constants['tcp_offset']])+offset
        hand_R=tool_R@R0.T;hand_P=tool_P-hand_R@relative_p
        state.update(tool_o=tool_P.tolist(),tool_R=tool_R.tolist())
        state['robots']['right'].update(TCP=hand_P.tolist(),R=hand_R.tolist())
    ctl.clock=(start+end)/2;observed_at(ctl.clock,np.array([.002,0.,0.]));before=ctl.clock
    ctl.act(observation())
    assert ctl.clock>before and not detail['actual_tool_placement']['phase_held']
    ctl.clock=end-.01;observed_at(end,np.array([.002,0.,0.]))
    ctl.act(observation(.10))
    assert ctl.clock==pytest.approx(end)
    assert not detail['actual_tool_placement']['open_mouth_boundary_ready']
    observed_at(end,np.zeros(3));ctl.act(observation(.15))
    assert ctl.clock>end and ctl.error is None
    __import__('json').dumps(detail)


def test_position_preload_requires_measured_full_opening_and_continues_smoothly_into_squeeze():
    ctl,state,detail=setup_controller();key=('right','beside the held cane');t=ctl.boundaries[key][0]
    ctl.clock=t-.01;ctl.open_reference=-.0446;ctl.open_good=3
    ctl.arrived.add(('right','open beside the cane'))
    p,R,_,_=ctl.policy._plans['right'].target(t);tool_R=R@np.diag([1.,-1.,-1.])
    tool_P=p-tool_R@np.array([0.,-.09177,.004])
    state.update(tool_o=tool_P.tolist(),tool_R=tool_R.tolist(),jaw=0.,handle_pad_contact_span_m=.0899)
    state['robots']['right'].update(TCP=p.tolist(),R=R.tolist(),fingers=[-.0446,-.0453])
    action=ctl.act(observation()).actions[0]
    assert ctl.preload_engaged
    assert detail['right']['finger_reference_m']==pytest.approx(-.0446+.00012)
    original_pose=state['tool_o'].copy()
    # Leaving the real open stop removes the preload; the existing entry gate still holds.
    state['jaw']=.001;ctl.clock=ctl.boundaries[('right','follow the open hook throat')][0]
    ctl.act(observation(.10))
    assert detail['right']['finger_reference_m']==pytest.approx(-.0446)
    assert detail['actual_tool_placement']['phase_held']
    # The ordinary squeeze starts from the preloaded reference rather than reopening first.
    ctl.clock=ctl.boundaries[('right','squeeze: cut')][0]
    state['jaw']=0.;ctl.act(observation(.15))
    begin,end=ctl.boundaries[('right','squeeze: cut')]
    expected=(-.0446+.00012)*(1-ik.smoothstep((ctl.clock-begin)/(end-begin)))
    assert detail['right']['finger_reference_m']==pytest.approx(expected)
    assert state['tool_o']==original_pose
    __import__('json').dumps(detail)


def test_accurate_entry_captures_a_reachable_stationary_hand_then_allows_real_jaw_closure(monkeypatch):
    ctl,state,detail=setup_controller();start=ctl.boundaries[('right','squeeze: cut')][0]
    ctl.clock=start-.01;ctl.open_reference=-.0446;ctl.preload_engaged=True
    ctl.arrived.update({('right','open beside the cane'),('right','beside the held cane'),
        ('right','follow the open hook throat'),('right','jaws round it, above the bud')})
    p,R,_,_=ctl.policy._plans['right'].target(start);relative_R=np.diag([1.,-1.,-1.]);tool_R=R@relative_R
    tool_P=p-tool_R@np.array([0.,-ctl.constants['grip_back'],ctl.constants['tcp_offset']])
    q,pe,re=ik.solve(p,ik.matrix_to_quat_xyzw(R),[.6,2.1,.9,1.2,-1.4,1.57],base_pos=station.BASES['right'],damping=.01,iters=400)
    assert pe<.0001 and re<np.radians(.1)
    relative_p=R.T@(tool_P-p);ctl.relative=(relative_p,relative_R)
    ctl.capture={'hand_to_tool_p_m':relative_p.tolist(),'hand_to_tool_R':relative_R.tolist()}
    state.update(tool_o=tool_P.tolist(),tool_R=tool_R.tolist(),jaw=0.,handle_pad_contact_span_m=.0899)
    state['robots']['right'].update(q=q.tolist(),TCP=p.tolist(),R=R.tolist(),fingers=[-.0446,-.0453])
    ctl.bias['right'][3]=.02;ctl.previous['right']=q+ctl.bias['right']
    ctl.act(observation())
    assert np.allclose(ctl.squeeze_capture['q'],q)
    ctl.policy._plans['left'].action7=lambda t:(np.r_[state['robots']['left']['q'],0.],'hold')
    monkeypatch.setattr(ik,'solve',lambda *args,**kw:(_ for _ in ()).throw(AssertionError('No new wrist IK during stationary squeeze')))
    for i in range(1,5):
        a=np.radians(5*i);rotation=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1.]])
        state['tool_R']=(tool_R@rotation).tolist();state['jaw']=.04*i
        action=ctl.act(observation(.05*(i+1))).actions[0]
        assert ctl.error is None and np.allclose(detail['right']['nominal'],q)
        assert np.allclose(action.data[7:13],q+ctl.bias['right'])
    assert not state['severed']  # only the unchanged native cutting law can establish this
    # Rotation during real closure is allowed; actual lost grip still stops it.
    state['both_handles']=False
    for i in range(103):
        ctl.act(observation(.30+.05*i))
        if ctl.error:break
    assert ctl.error=='Timeout: actual tool placement tracking'
    __import__('json').dumps(detail)


def test_controller_can_complete_when_physical_observations_confirm_each_phase():
    """Interface/clock replay, not a simulated contact or task-score proof."""
    ctl,state,_=setup_controller()
    close_end=ctl.boundaries[('left','hold it')][1]
    opening=ctl.boundaries[('right','open beside the cane')][0]
    recenter=ctl.boundaries[('right','recenter vertically')][0]
    cut_end=ctl.boundaries[('right','squeeze: cut')][1]
    release=ctl.boundaries[('right','let go')][1]
    for step in range(1100):
        state['both_target_fingers']=ctl.clock>=close_end-.05
        state['severed']=ctl.clock>=cut_end-.05
        state['both_handles']=ctl.clock<release-.05
        if ctl.clock<recenter:state['jaw']=.51
        elif ctl.clock<opening:state['jaw']=.45
        else:state['jaw']=max(0.,(.0464+ctl.open_reference)*18)
        r=state['robots']['right'];R=np.asarray(r['R']);P=np.asarray(r['TCP'])
        tool_R=R@np.diag([1.,-1.,-1.]);tool_P=P-tool_R@np.array([0.,-.09,.006])
        state.update(tool_o=tool_P.tolist(),tool_R=tool_R.tolist())
        action=ctl.act(observation(.05*(step+1))).actions[0]
        for side,offset in(('left',0),('right',7)):
            q=np.asarray(action.data[offset:offset+6]);P,R=ik.tcp_pose(q,station.BASES[side])
            state['robots'][side].update(q=q.tolist(),TCP=P.tolist(),R=R.tolist(),
                fingers=[float(action.data[offset+6]*__import__('yam_arena.constants',fromlist=['FINGER_OPEN']).FINGER_OPEN)]*2)
        state['handle_pad_contact_span_m']=abs(sum(state['robots']['right']['fingers']))
        if action.meta.get('request_stop'):break
    assert ctl.error is None
    assert ctl.clock>ctl.policy._end


def test_live_supported_pitch_keeps_a_reachable_hand_instead_of_chasing_the_hanging_tool(monkeypatch):
    ctl,state,detail=setup_controller();key=('right','beside the held cane');start,end=ctl.boundaries[key]
    ctl.clock=(start+end)/2;ctl.arrived.update({key,('right','open beside the cane')})
    p,R,_,_=ctl.policy._plans['right'].target(ctl.clock)
    nominal=R@np.diag([1.,-1.,-1.]);tool_R=nominal@ctl.sa._rx(-.1)
    offset=np.array([0.,-ctl.constants['grip_back'],ctl.constants['tcp_offset']])
    nominal_P=p-nominal@offset;section=nominal.T@(ctl.material_points['squeeze: cut']-nominal_P)
    tool_P=ctl.material_points['squeeze: cut']-tool_R@section
    relative_p=R.T@(tool_P-p);relative_R=R.T@tool_R
    ctl.relative=(relative_p,relative_R);ctl.capture={'hand_to_tool_p_m':relative_p.tolist(),'hand_to_tool_R':relative_R.tolist()}
    state['robots']['right'].update(TCP=p.tolist(),R=R.tolist(),fingers=[-.0446,-.0453])
    state.update(tool_o=tool_P.tolist(),tool_R=tool_R.tolist(),jaw=0.,handle_pad_contact_span_m=.0899)
    goals=[]
    monkeypatch.setattr(ik,'solve',lambda p,q,seed,**kw:(goals.append(np.array(q))or(np.array(seed),0.,0.)))
    ctl.act(observation())
    assert ctl.error is None
    assert abs(goals[-1]@ik.matrix_to_quat_xyzw(R))==pytest.approx(1.)
    assert detail['actual_tool_placement']['rotation_error_rad']<1e-7
    assert detail['actual_tool_placement']['raw_planned_rotation_error_rad']==pytest.approx(.1)
    __import__('json').dumps(detail)
