"""Bounded, state-aware scripted pruning using ordinary position actions only."""
from __future__ import annotations

import numpy as np
from yam_arena import ik
from yam_arena.constants import FINGER_OPEN,READY
from supported_pitch import supported_pose,OpenThroat


def bounded_position(nominal,actual,velocity,gravity,K,D,effort,limits,speed,previous,bias,dt):
    """Intersect position, rate and static PD+GC effort-reference intervals.

    Native effort limits remain authoritative. This reference estimate is not
    a measurement of total contact or motor torque.
    """
    nominal,actual,velocity,gravity,K,D,effort,speed,previous,bias=map(np.asarray,
        (nominal,actual,velocity,gravity,K,D,effort,speed,previous,bias))
    limits=np.asarray(limits)
    if not all(np.isfinite(v).all()for v in (nominal,actual,velocity,gravity,K,D,effort,limits,speed,previous,bias)) or np.any(K<=0):
        raise ValueError('Unknown/nonfinite position-reference input')
    low=np.maximum(limits[:,0],previous-speed*dt)
    high=np.minimum(limits[:,1],previous+speed*dt)
    low=np.maximum(low,actual+(-effort+D*velocity-gravity)/K)
    high=np.minimum(high,actual+(effort+D*velocity-gravity)/K)
    if np.any(low>high):raise ValueError('Position/rate/effort reference intersection is empty')
    delta=np.clip(.5*(nominal-actual)*dt,-.004*dt/.05,.004*dt/.05)
    candidate=np.clip(bias+delta,-.12,.12)
    requested=nominal+candidate
    sent=np.clip(requested,low,high)
    outward=((requested>high)&(delta>0))|((requested<low)&(delta<0))
    candidate=np.where(outward,bias,candidate)
    sent=np.clip(nominal+candidate,low,high)
    return sent,candidate


class PruningController:
    """The script waits for physical grasp/arrival; it cannot authorize a cut."""
    def __init__(self,policy,snapshot,diagnostics,sa,task_constants,mode='success'):
        self.policy,self.snapshot,self.diagnostics=policy,snapshot,diagnostics
        self.sa,self.constants,self.mode=sa,task_constants,mode
        self.clock=0.;self.last_time=None;self.gate_wait=0.;self.gate_name=None
        self.bias={s:np.zeros(6)for s in policy._plans}
        self.previous={s:np.array(READY,float)for s in policy._plans}
        self.error=None;self.relative=None;self.capture=None;self.open_reference=-.021
        self.open_good=0;self.arrived=set();self.last_open_observation=None
        self.placement_wait=0.
        self.preload_engaged=False
        self.squeeze_capture=None
        self.open_throat=OpenThroat(sa)
        self.material_points={}
        offset=np.array([0.,-self.constants['grip_back'],self.constants['tcp_offset']])
        section=np.asarray(self.constants['cut_section'])
        for w in policy._plans['right'].waypoints:
            if w.label in ('squeeze: cut','second cut'):
                R=np.asarray(w.rotation)@np.diag([1.,-1.,-1.])
                self.material_points[w.label]=np.asarray(w.pos)+R@(section-offset)
        self.boundaries={}
        for side,plan in policy._plans.items():
            t=0.
            for w in plan.waypoints:
                self.boundaries[(side,w.label)]=(t,t+w.duration);t+=w.duration

    def _gate(self,name,good,dt,timeout=5.):
        if good:
            self.arrived.add(name);self.gate_wait=0.;self.gate_name=None
            return True
        if self.gate_name!=name:self.gate_name=name;self.gate_wait=0.
        self.gate_wait+=dt
        if self.gate_wait>timeout:self.error=f'Timeout: {name}'
        return False

    def _at_boundary(self,key,next_clock):
        if key not in self.boundaries:return False
        t=self.boundaries[key][0]
        return key not in self.arrived and self.clock<=t<next_clock+1e-9

    def _hold_boundary(self,key,next_clock,good,dt,timeout=5.):
        if self._at_boundary(key,next_clock) and not self._gate(key,good,dt,timeout):
            return self.boundaries[key][0]
        return next_clock

    def _supported_target(self,pos,hand_R,label,state):
        R=np.asarray(hand_R)@np.diag([1.,-1.,-1.])
        P=np.asarray(pos)-R@np.array([0.,-self.constants['grip_back'],self.constants['tcp_offset']])
        which='second cut'if label in ('follow the retained hook throat','jaws round the retained plant','second cut')else'squeeze: cut'
        return supported_pose(P,R,self.material_points[which],state['tool_R'])

    def _observed_open(self,state):
        hand=state['robots']['right'];fingers=np.asarray(hand['fingers'])
        limits=np.asarray(hand['finger_limits']);span=state.get('handle_pad_contact_span_m')
        hardware_span=-float(limits[:,0].sum());gap=abs(float(fingers.sum()))
        return (state['jaw']<=.0005 and state['both_handles'] and span is not None
                and np.all(fingers>=limits[:,0]-1e-6) and np.all(fingers<=limits[:,1]+1e-6)
                and gap<=hardware_span+1e-6 and abs(float(span)-gap)<=.001)

    def act(self,observation):
        from inspect_robots.types import Action,ActionChunk
        from inspect_robots_yam.packing import pack
        now=float(observation.state_time)
        dt=.05 if self.last_time is None else float(np.clip(now-self.last_time,.001,.1))
        self.last_time=now
        state=self.snapshot()
        if state is None:
            return ActionChunk(actions=[Action(data=pack(np.r_[READY,1.],np.r_[READY,1.]))],control_hz=20.)
        next_clock=self.clock+dt
        robots=state['robots']
        if now>58:self.error='Bounded demonstration timeout'
        if state['blocking_left'] and self.clock<self.boundaries.get(('left','draw it out'),(1e9,))[0]:
            self.error='Unexpected left-arm/plant or fixture contact'
        right_label=self.policy._plans['right'].target(self.clock)[3]
        table_paths=state.get('robot_contact_detail',{}).get('table_paths',[])
        allowed_table=[];blocked_right=[]
        for row in state.get('blocking_right',[]):
            f=np.asarray(row['force']);point=np.asarray(row['point'])
            support=(right_label in ('round the handles','close for carrying','lift closed','lay them down','let go','rise')
                and row['sensor'].rsplit('/',1)[-1]in ('tip_left','tip_right')
                and row['other']in table_paths and f[2]>.99*np.linalg.norm(f)
                and abs(point[2]-self.constants['table_top'])<=.0005 and row['separation']>=-.0005)
            (allowed_table if support else blocked_right).append(row)
        if blocked_right:
            self.error='Unexpected right-arm/plant or fixture contact'
        if self.mode!='green' and state.get('wrong_tool_plant_contacts'):
            self.error='Tool contacted a non-target plant surface'
        # A real open grasp can creep. Update only an observed bilateral grasp;
        # moving the hand to place the measured tool is ordinary position control.
        pause_right=False;placement=None
        live_labels=('beside the held cane','follow the open hook throat',
                     'jaws round it, above the bud','squeeze: cut','hold','back out closed','to a free spot')
        if self.relative is not None:
            plan=self.policy._plans['right'];pos,R,_,label=plan.target(self.clock)
            if label in live_labels:
                hand=robots['right'];P=np.asarray(hand['TCP']);H=np.asarray(hand['R'])
                toolP,toolR=np.asarray(state['tool_o']),np.asarray(state['tool_R'])
                measured=(H.T@(toolP-P),H.T@toolR)
                # Closing the real handles changes the axial contact relation.
                # The open-entry envelope remains tied to its opening capture;
                # closing/closed handling instead checks each observed update.
                closing=label in ('squeeze: cut','hold','back out closed','to a free spot')
                original_p=(self.relative[0] if closing else np.asarray(self.capture['hand_to_tool_p_m']))
                original_R=(self.relative[1] if closing else np.asarray(self.capture['hand_to_tool_R']))
                # Closing uses a captured hand posture, not this placement transform.
                # Real bilateral finite contact remains required in that phase.
                # Grip location is measured in the tool, not a rotating hand.
                # Rolling contact may rotate a long hand-frame lever without
                # sliding the fingers along the actual handles.
                grip_in_tool=-measured[1].T@measured[0]
                reference_in_tool=-original_R.T@original_p
                shift=float(np.linalg.norm(grip_in_tool-reference_in_tool))
                twist=float(np.arccos(np.clip(measured[1][:,0]@self.relative[1][:,0],-1.,1.)))
                total_twist=float(np.arccos(np.clip((np.trace(measured[1].T@np.asarray(self.capture['hand_to_tool_R']))-1)/2,-1.,1.)))
                if not closing and (shift>.006 or twist>np.radians(3)):
                    self.error='Working grasp moved outside the bounded placement reference'
                elif state['both_handles']:
                    self.relative=measured
                else:pause_right=True
                goalP,goalR,free_pitch,rotation_error=self._supported_target(pos,R,label,state)
                head=np.array([0.,.057,0.])
                position_error=max(np.linalg.norm(toolP-goalP),np.linalg.norm(toolP+toolR@head-goalP-goalR@head))
                raw_rotation_error=float(np.arccos(np.clip((np.trace(toolR.T@(R@np.diag([1.,-1.,-1.])))-1)/2,-1.,1.)))
                tracked=bool(state['both_handles'] and position_error<=.001 and rotation_error<=np.radians(.5))
                open_for_entry=self._observed_open(state)
                need_hold=(pause_right or (label in live_labels[1:3] and not tracked)
                    or (label in live_labels[1:3] and not open_for_entry))
                placement={'label':label,'position_error_m':float(position_error),
                    'rotation_error_rad':rotation_error,'supported_pitch_rad':free_pitch,
                    'raw_planned_rotation_error_rad':raw_rotation_error,'grasp_shift_m':shift,'grasp_rotation_rad':twist,
                    'grasp_reference':'previous_bilateral_closing' if closing else 'original_open_capture',
                    'rotation_from_open_capture_rad':total_twist,'TCP_in_tool_m':grip_in_tool.tolist(),
                    'actual_tool_P':toolP.tolist(),'actual_tool_R':toolR.tolist(),
                    'desired_tool_P':goalP.tolist(),'desired_tool_R':goalR.tolist(),
                    'bilateral_handle_pressure':bool(state['both_handles']),'within_tracking_tolerance':tracked,
                    'actual_open_for_entry':bool(open_for_entry),'phase_held':need_hold}
                if placement['phase_held']:
                    next_clock=self.clock;self.placement_wait+=dt
                    if self.placement_wait>5.:self.error='Timeout: actual tool placement tracking'
                else:self.placement_wait=0.
                # The cleared free-space prefix may track while moving. At its
                # open-mouth endpoint, require the precise controlled condition before
                # advancing into the finite hook throat.
                pre_key=('right','follow the open hook throat')
                if self._at_boundary(pre_key,next_clock):
                    p0,R0,_,_=plan.target(self.boundaries[pre_key][0])
                    preP,preR,_,pre_angle=self._supported_target(p0,R0,'beside the held cane',state)
                    pre_error=max(np.linalg.norm(toolP-preP),np.linalg.norm(toolP+toolR@head-preP-preR@head))
                    ready=open_for_entry and pre_error<=.001 and pre_angle<=np.radians(.5)
                    next_clock=self._hold_boundary(pre_key,next_clock,ready,dt,5.)
                    placement['open_mouth_boundary_ready']=bool(ready)
                    if not ready:placement['phase_held']=True
        if 'left' in self.policy._plans:
            key=('left','hold it')
            plan=self.policy._plans['left'];p,R,_,_=plan.target(self.boundaries[key][0])
            actual=robots['left']
            arrived=np.linalg.norm(np.asarray(actual['TCP'])-p)<.003 and np.linalg.norm(ik._rot_error(R,np.asarray(actual['R'])))<np.radians(3)
            next_clock=self._hold_boundary(key,next_clock,arrived,dt)
        # Do not relax the tool before the other hand has really acquired the target.
        next_clock=self._hold_boundary(('right','recenter vertically'),next_clock,state['both_target_fingers'],dt)
        next_clock=self._hold_boundary(('right','open beside the cane'),next_clock,state['both_target_fingers'],dt)
        key=('right','beside the held cane')
        if self._at_boundary(key,next_clock):
            item=robots['right'];P,R=np.asarray(item['TCP']),np.asarray(item['R'])
            toolP,toolR=np.asarray(state['tool_o']),np.asarray(state['tool_R'])
            local=toolR.T@(P-toolP)
            # Full-opening feasibility is observed, not extrapolated from a
            # fixed near-closed axial grip. The handles can move as they open.
            centred=abs(local[0])<.003 and self.open_good>=3 and self._observed_open(state)
            if self._gate(key,centred,dt,5.):
                self.relative=(R.T@(toolP-P),R.T@toolR)
                self.capture={'observation_time_s':now,'physics_counter':state.get('physics_counter'),
                    'hand_to_tool_p_m':self.relative[0].tolist(),'hand_to_tool_R':self.relative[1].tolist(),
                    'actual_jaw_rad':state['jaw'],'actual_fingers_m':item['fingers']}
                self.preload_engaged=True
            else:next_clock=self.boundaries[key][0]
        # Opening is slow and stops at the actual opening stop, rather than
        # opening past the handles and throwing away their pressure.
        opening=self.boundaries[('right','open beside the cane')]
        if ('right','open beside the cane') in self.arrived and ('right','beside the held cane') not in self.arrived:
            if state['jaw']>.0005:
                self.open_reference=max(FINGER_OPEN,self.open_reference-.008*dt)
                self.open_good=0
            elif self._observed_open(state):
                self.open_good+=1
            else:self.open_good=0
        entry_ready=(self.relative is not None and self._observed_open(state)
                     and state['both_target_fingers'])
        next_clock=self._hold_boundary(('right','jaws round it, above the bud'),next_clock,entry_ready,dt)
        key=('right','squeeze: cut')
        ready=state['both_target_fingers'] and state['both_handles']
        if self._at_boundary(key,next_clock):
            plan=self.policy._plans['right'];p0,R0,_,_=plan.target(self.boundaries[key][0])
            goalP,goalR,_,re=self._supported_target(p0,R0,'jaws round it, above the bud',state)
            actualP,actualR=np.asarray(state['tool_o']),np.asarray(state['tool_R'])
            head=np.array([0.,.057,0.])
            pe=max(np.linalg.norm(actualP-goalP),np.linalg.norm(actualP+actualR@head-goalP-goalR@head))
            ready=bool(ready and self._observed_open(state) and pe<=.001 and re<=np.radians(.5))
            if ready:
                hand=robots['right']
                self.squeeze_capture={'q':list(hand['q']),'TCP':list(hand['TCP']),'R':hand['R'],
                    'nominal_TCP':p0.tolist(),'physics_counter':state.get('physics_counter'),
                    'entry_position_error_m':float(pe),'entry_rotation_error_rad':re}
        next_clock=self._hold_boundary(key,next_clock,ready,dt)
        # An intact stem must not be pulled out as if it had already severed.
        if self.mode!='green':
            next_clock=self._hold_boundary(('left','draw it out'),next_clock,state['severed'],dt,4.)
        flat=(abs(np.asarray(state['tool_R'])[2,2])>np.cos(np.radians(20))
              and state['tool_o'][2]<self.constants['table_top']+.021)
        next_clock=self._hold_boundary(('right','let go'),next_clock,flat,dt,3.)
        released=(not state['both_handles'] and abs(sum(robots['right']['fingers']))>.080)
        next_clock=self._hold_boundary(('right','rise'),next_clock,released,dt,3.)
        self.clock=next_clock
        outputs={};detail={'clock':self.clock,'gate':str(self.gate_name),'wait_s':self.gate_wait,
                           'observation_time_s':now,'physics_counter':state.get('physics_counter'),
                           'open_reference_m':self.open_reference,'error':self.error}
        detail['actual_tool_to_TCP_midframe_m']=state.get('actual_tool_to_TCP_midframe_m')
        detail['post_open_grasp_capture']=self.capture
        detail['squeeze_hand_capture']=self.squeeze_capture
        detail['handle_pad_contact_span_m']=state.get('handle_pad_contact_span_m')
        detail['actual_finger_gap_m']=abs(sum(robots['right']['fingers']))
        detail['expected_table_tip_support']=allowed_table
        detail['blocking_right_contacts']=blocked_right
        detail['wrong_tool_plant_contacts']=state.get('wrong_tool_plant_contacts',[])
        detail['blocking_left_contacts']=state['blocking_left']
        detail['payload_transport_contacts']=state.get('payload_transport_contacts',[])
        detail['current_handle_contact_checks']=state.get('handle_contact_checks',[])
        detail['current_tool_plant_contacts']=state.get('tool_plant_contact_points',[])
        detail['current_plant_body_poses']=state.get('plant_body_poses',{})
        detail['actual_tool_placement']=placement
        detail['live_hand_to_tool_reference']=(None if self.relative is None else
            {'position_m':self.relative[0].tolist(),'rotation':self.relative[1].tolist()})
        for side in ('left','right'):
            if side not in self.policy._plans:
                outputs[side]=np.r_[READY,1.];continue
            plan=self.policy._plans[side]
            pos,R,finger,label=plan.target(self.clock)
            stationary=(side=='right' and self.squeeze_capture is not None and label in ('squeeze: cut','hold'))
            if stationary:
                nominal=np.asarray(self.squeeze_capture['q']);plan.q=nominal.copy()
            else:
                action,_=plan.action7(self.clock);nominal=action[:6]
            if side=='right':
                if label in ('open beside the cane','beside the held cane','follow the open hook throat','follow the retained hook throat','jaws round it, above the bud','wait for the holding hand'):
                    finger=self.open_reference if ('right','open beside the cane')in self.arrived else -.021
                    if self.preload_engaged and state['jaw']<=.0005:
                        finger+=.00012  # ordinary position preload; real open-stop/contact gates stay unchanged
                elif label in ('squeeze: cut','second cut'):
                    start,end=self.boundaries[(side,label)]
                    start_reference=self.open_reference+(.00012 if self.preload_engaged else 0.)
                    finger=start_reference*(1-ik.smoothstep((self.clock-start)/(end-start)))
                # After recentering, place the finite tool with its measured
                # hand-relative transform, not a nominal symmetric grasp.
                if self.squeeze_capture is not None and label in ('back out closed','to a free spot'):
                    from pose_motion import closed_hand_pose
                    desired_P,desired_R=closed_hand_pose(self.squeeze_capture,label,self.clock,self.boundaries,pos,R)
                    nominal,pe,re=ik.solve(desired_P,ik.matrix_to_quat_xyzw(desired_R),plan.q,
                        base_pos=plan.base_pos,iters=100,tol=1e-5)
                    if pe>.001 or re>np.radians(1):self.error='Closed-hand withdrawal IK failed'
                    plan.q=nominal
                elif not stationary and self.relative is not None and label in ('beside the held cane','open beside the cane','follow the open hook throat','follow the retained hook throat','jaws round it, above the bud','squeeze: cut','hold','back out closed','to a free spot'):
                    relative_p,relative_R=self.relative
                    tool_P,tool_R,pitch,residual=self._supported_target(pos,R,label,state)
                    desired_R=tool_R@relative_R.T
                    desired_P=tool_P-desired_R@relative_p
                    if not state['severed'] and label not in ('squeeze: cut','hold','back out closed','to a free spot'):
                        try:
                            finite=self.open_throat(state,tool_P,tool_R)
                            detail['predicted_open_throat']=finite
                            if finite['gap_lower_bound_m']<0:
                                self.error='Supported pitch has no finite open throat'
                        except ValueError as exc:
                            self.error=str(exc);detail['predicted_open_throat']={'unknown':str(exc)}
                    if self.error is None:
                        nominal,pe,re=ik.solve(desired_P,ik.matrix_to_quat_xyzw(desired_R),plan.q,
                            base_pos=plan.base_pos,iters=100,tol=1e-5)
                        if pe>.001 or re>np.radians(1):self.error='Actual-grasp placement IK failed'
                        plan.q=nominal
            actual=robots[side]
            try:
                # Blocking contacts do not receive an increasing bias.
                if side=='left' and state['blocking_left']:raise ValueError('Blocked approach')
                if side=='right' and pause_right:
                    sent,bias=np.asarray(actual['q']),np.zeros(6)
                else:
                    sent,bias=bounded_position(nominal,actual['q'],actual['qd'],actual['gravity_Nm'],
                        actual['K'],actual['D'],actual['effort'],actual['limits'],actual['speed'],
                        self.previous[side],self.bias[side],dt)
                self.bias[side]=bias
            except ValueError as exc:
                self.error=str(exc);sent=np.asarray(actual['q'])
            self.previous[side]=sent
            outputs[side]=np.r_[sent,np.clip(finger/FINGER_OPEN,0.,1.)]
            detail[side]={'label':label,'nominal':nominal.tolist(),'sent':sent.tolist(),
                          'bias':self.bias[side].tolist(),'actual':actual['q'],'finger_reference_m':float(finger)}
        detail['error']=self.error
        if self.error:
            # Stop a blocked approach at the measured robot pose. This is an
            # ordinary final position target, never a state correction.
            for side in self.policy._plans:
                actual=robots[side]
                outputs[side]=np.r_[actual['q'],np.clip(np.mean(actual['fingers'])/FINGER_OPEN,0.,1.)]
                detail[side]['finger_reference_m']=float(np.mean(actual['fingers']))
            detail['aborted_position_hold']=True
        self.diagnostics(detail)
        finished=self.clock>self.policy._end+self.policy.hold_s
        meta={'request_stop':True,'stop_reason':self.error or 'pruning_plan_complete'}if self.error or finished else {}
        return ActionChunk(actions=[Action(data=pack(outputs['left'],outputs['right']),meta=meta)],control_hz=20.)
