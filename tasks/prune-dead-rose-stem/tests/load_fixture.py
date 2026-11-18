"""Synthetic measured-contact inputs for scorer geometry tests, without a simulator."""
from types import SimpleNamespace
import numpy as np
import task


def loaded_tool():
    return SimpleNamespace(joint_names=['pivot', 'fuse'], data=SimpleNamespace(joint_pos=np.zeros((1, 2))))


def set_blade_angle(tool, angle):
    lag = -.12 if angle > .1 else 0.
    tool.data.joint_pos[0] = [angle-lag, lag]


def measured_contacts(env):
    """Give each specified cane a current normal load; geometry still decides a cut."""
    tool, bush = env.unwrapped.scene['secateurs'], env.unwrapped.scene['bush']
    if task._tool_joint(tool, 'fuse') == 0:
        return []
    o, r = task._body(tool, 'hook_half', np.zeros(3))
    shift = np.array([0., 0., task.TOP]);rows=[]
    for name,a,b,s_min in task._canes(bush,np.zeros(3)):
        if name.startswith('bud') or(name=='dead'and task._S['cut']):continue
        hit=task._jaw_point(a,b,o,r,s_min)
        if hit is None:continue
        point=o+r@hit[0]+shift
        rows.append({'body':'pot'if name.startswith('green')else'stub_col',
                     'point_world_m':point.tolist(),'force_on_blade_N':(r[:,0]*2*task.F_CUT).tolist(),
                     'separation_m':0.,'sensor':'/tool/blade','plant_body':'/plant'})
    return rows


def attach(env):
    task._S['contacts']=SimpleNamespace(sample=lambda:measured_contacts(env), sample_handles=lambda:[])
