from types import SimpleNamespace
import numpy as np
import pytest
from pruning_observation import MotionReader


class ContactView:
    def __init__(self,mismatch=False):self.mismatch=mismatch
    def get_contact_data(self,dt):
        return (np.array([[-1.]]),np.array([[0.,0.,.275]]),np.array([[1.,0.,0.]]),
                np.zeros((1,1)),np.ones((1,1),int),np.zeros((1,1),int))
    def get_contact_force_matrix(self,dt):return np.array([[[-1.+self.mismatch,0.,0.]]])
    def get_net_contact_forces(self,dt):return np.array([[-1.,0.,0.]])
    def get_friction_data(self,dt):raise RuntimeError('unavailable in this test backend')


def test_signed_native_contact_is_preserved_and_unknown_friction_is_not_zero():
    reader=MotionReader.__new__(MotionReader);reader.dt=1/240
    reader.sensors=['/LeftArm/tip_left'];reader.filters=['/bush/cut_top'];reader.table_paths=[]
    reader.view=ContactView()
    rows,detail=reader.contacts(np.zeros(3))
    assert rows[0]['signed_normal_N']==-1
    assert rows[0]['force']==[-1.,0.,0.]
    assert 'unknown'in detail['friction_points']
    reader.view=ContactView(mismatch=True)
    with pytest.raises(RuntimeError,match='current-step pair'):reader.contacts(np.zeros(3))


def pose_array(n,z=.14):
    p=np.zeros((1,n,7));p[:,:,2]=z;p[:,:,6]=1.;return p


def target_reader(points):
    reader=MotionReader.__new__(MotionReader)
    reader.sa=SimpleNamespace(R0=.00225)
    robot=SimpleNamespace(joint_names=[f'joint{i}'for i in range(1,9)],body_names=['gripper','tip_left','tip_right'],
        data=SimpleNamespace(joint_pos=np.zeros((1,8)),joint_vel=np.zeros((1,8)),
             body_link_pose_w=pose_array(3),gravity_compensation_forces=np.zeros((1,8))),
        root_physx_view=SimpleNamespace())
    robot.data.joint_pos[0,6:]=-.00225
    tool=SimpleNamespace(body_names=['hook_half','blade_half','blade'],joint_names=['pivot','fuse'],
        data=SimpleNamespace(joint_pos=np.array([[.51,0.]]),body_link_pose_w=pose_array(3)))
    bush=SimpleNamespace(body_names=['pot','cut_top'],data=SimpleNamespace(body_link_pose_w=pose_array(2)))
    reader.base=SimpleNamespace(scene={'left_robot':robot,'right_robot':robot,'secateurs':tool,'bush':bush},
                               sim=SimpleNamespace(get_physics_step_count=lambda:123))
    reader.robot_limits={s:{}for s in('left','right')}
    reader.finger_limits={s:[[-.04695,0.],[-.04695,0.]]for s in('left','right')}
    reader.contacts=lambda origin,*args:(points,{'friction_points':{'unknown':'test'}})
    reader.tool_plant_view=object();reader.tool_sensors=['/secateurs/hook_half'];reader.plant_filters=['/bush/pot']
    reader.pads=SimpleNamespace(contains_surface=lambda name,p:True)
    return reader


def pressure(finger,sign,z=.275,body='cut_top'):
    return {'sensor':f'/World/envs/env_0/LeftArm/{finger}','other':f'/bush/{body}',
            'point':[sign*.00225,0.,z],'force':[sign,0.,0.]}


def test_true_opposing_inner_pad_contacts_acquire_the_actual_target():
    reader=target_reader([pressure('tip_left',-1),pressure('tip_right',1)])
    state=reader.sample(np.zeros(3),[0,0,0],[0,0,.31],[],False)
    assert state['both_target_fingers']
    assert not state['blocking_left']
    assert state['physics_counter']==123
    assert state['plant_body_poses']['pot']['rotation']==np.eye(3).tolist()
    assert state['tool_plant_contact_points']


def test_tool_plant_diagnostic_preserves_real_pair_identity_without_work_domain_crop():
    reader=MotionReader.__new__(MotionReader);reader.dt=1/240;reader.table_paths=[]
    reader.view=None;reader.sensors=['unused'];reader.filters=['unused']
    rows,detail=reader.contacts(np.zeros(3),ContactView(),['/secateurs/hook_half'],['/bush/pot'])
    assert rows[0]['sensor']=='/secateurs/hook_half'
    assert rows[0]['other']=='/bush/pot'
    assert rows[0]['force']==[-1.,0.,0.]
    assert detail['sensor_paths']==['/secateurs/hook_half']
    assert 'unknown'in detail['friction_points']


@pytest.mark.parametrize('bad',[
    pressure('tip_right',-1),pressure('tip_right',1,z=.309),pressure('tip_right',1,body='pot')])
def test_wrong_normal_tip_obstruction_and_wrong_plant_body_are_not_a_grasp(bad):
    reader=target_reader([pressure('tip_left',-1),bad])
    state=reader.sample(np.zeros(3),[0,0,0],[0,0,.31],[],False)
    assert not state['both_target_fingers']
    assert state['blocking_left']


def test_own_severed_payload_side_contact_is_support_not_a_new_grasp_or_block():
    # Same native pair/force as the observed transport stop; it is the actual
    # cutting on a real tip, not a neighbouring cane or a housing obstruction.
    row={'sensor':'/World/envs/env_0/LeftArm/tip_right','other':'/bush/cut_top',
         'point':[.2245964557,.2999088466,.9750753641],
         'force':[-8.3889007568,-5.7284560204,.2634294927]}
    reader=target_reader([row]);state=reader.sample(np.zeros(3),[0,0,0],[0,0,.31],[],True)
    assert not state['blocking_left'] and not state['both_target_fingers']
    assert state['payload_transport_contacts']==[row]
    for changed in ({'other':'/bush/pot'},{'sensor':'/World/envs/env_0/LeftArm/gripper'}):
        bad={**row,**changed};reader=target_reader([bad])
        state=reader.sample(np.zeros(3),[0,0,0],[0,0,.31],[],True)
        assert state['blocking_left']==[bad]
        assert not state['payload_transport_contacts']
