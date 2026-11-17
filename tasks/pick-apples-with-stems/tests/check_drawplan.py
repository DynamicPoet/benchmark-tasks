"""Physical-frame and fail-closed contracts for ordinary load references."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import drawplan
from yam_arena import ik
from yam_arena.station import BASES

gains=np.array([80,80,80,10,10,10.]);limits=np.c_[ik.LOWER,ik.UPPER]
efforts=np.array([27,27,27,7,7,7.]);vel=np.array([5.5,5.5,5.5,20,20,20.])
poses=[(np.array([.8033,1.3025,1.4869,-.6876,.9095,.9608]),np.array([.171,.470,-.866])),
       (np.array([.7831,1.7118,2.2217,-1.5216,.2839,-.6151]),np.array([.470,-.171,-.866]))]
for q,u in poses:
 target,r=drawplan.reference(q,q,u,gains,limits,efforts,vel,BASES['right'])
 assert (target>=ik.LOWER).all()and(target<=ik.UPPER).all()
 force=np.array(r['intended_TCP_force_N']);tau=np.array(r['intended_extra_joint_torque_Nm'])
 assert abs(force@u/np.linalg.norm(u)+22)<1e-12  # away from the real seat
 # Independent small-motion virtual work uses FK positions, not a second
 # force command or an assumed applied 22 N.
 dq=np.array([.2,-.1,.3,.4,-.2,.1])*1e-7
 dx=ik.tcp_pose(q+dq,BASES['right'])[0]-ik.tcp_pose(q-dq,BASES['right'])[0]
 assert abs(force@dx-2*tau@dq)<1e-12
 zero,_=drawplan.reference(q,q,u,gains,limits,efforts,vel,BASES['right'],force_N=0)
 assert np.array_equal(zero,q)
 try:drawplan.reference(q,q,u,gains,limits,efforts,vel,BASES['right'],force_N=22.001)
 except ValueError:pass
 else:raise AssertionError('The intended reference must stay within its declared bound')
 for bad_k,bad_ramp,bad_effort in [(gains*0,4,efforts),(gains*.0001,4,efforts),(gains,.0001,efforts),(gains,4,efforts*.0001)]:
  try:drawplan.reference(q,q,u,bad_k,limits,bad_effort,vel,BASES['right'],ramp_s=bad_ramp)
  except ValueError:pass
  else:raise AssertionError('Unknown/infeasible references must fail, never silently clip')
print('Axial position-reference contracts passed')
