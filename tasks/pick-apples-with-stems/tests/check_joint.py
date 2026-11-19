"""Constitutive adversaries, without a simulator or object-state writes."""
from pathlib import Path
import sys,math
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import stemjoint as j

# Straight tension must pull the peg, not claim a clean seat release.
assert j.part_of(dict(tension=23.99,bending=0.0)) is None
assert j.part_of(dict(tension=24.,bending=0.0)) == 'stem'
# A combined sub-peg draw and upward bend can tip the seat.
assert j.part_of(dict(tension=17.,bending=.15)) == 'seat'
assert j.part_of(dict(tension=0.,bending=j.tip_moment())) == 'seat'
# Compression and cap twist do not masquerade as stem tension or bending.
assert j.part_of(dict(tension=-100.,bending=0.,twist=10.)) is None
assert abs(j.tip_moment()-.254)<1e-12
assert abs(j.twist_limit(0.)-.0423333333333333)<1e-12
# Both thresholds pass in one substep, in either order. Final overload
# magnitude alone picks the wrong interface in each of the first two cases.
assert j.first_crossing((.99, .75), (1.05, 1.15)) == "seat"
assert j.first_crossing((.75, .99), (1.15, 1.05)) == "stem"
assert j.first_crossing((.9, .8), (1.1, 1.2)) == "stem"
# A rigid rotation of the whole rig preserves the spring wrench and loads.
from yam_arena.urdf import axis_angle_matrix as R
axis=np.array([.2,.6,.7]);axis/=np.linalg.norm(axis);A=R(axis,.9)
g=np.array([0.,0.,-9.81]);home=np.eye(3);anchor=np.array([.0445,0.,0.]);c=np.zeros(3);p=anchor.copy()
def state(B):
 q=j.Joint('x',anchor,p,home,.18,.4*.18*.0375**2);q.settle(B@g,B)
 # A common world rigid rotation is applied to every pose and gravity.
 F=B@R(np.array([0.,1.,0.]),.2)
 return q.wrench(B@c,F,np.zeros(3),np.zeros(3),np.zeros(3),B,np.zeros(3),np.zeros(3),np.zeros(3),1/240)
f,m,_,loads=state(np.eye(3));fa,ma,_,la=state(A)
assert np.allclose(fa,A@f,atol=1e-10)
assert np.allclose(ma,A@m,atol=1e-10)
assert all(abs(loads[k]-la[k])<1e-10 for k in loads)
# Repeated twist slips on the magnetic cap without generating a clean-pick
# criterion: check the actual reported wrench after rest-slip, not a motion tag.
q=j.Joint('x',anchor,p,home,.18,.4*.18*.0375**2);q.settle(g,np.eye(3))
for angle in np.linspace(0,2.5,100):
 F=j.rot_x(angle)
 _,_,_,loads=q.wrench(c,F,np.zeros(3),np.zeros(3),np.zeros(3),home,np.zeros(3),np.zeros(3),np.zeros(3),1/240)
 assert loads['twist']<=j.twist_limit(loads['tension'])+1e-10
 assert j.part_of(loads) is None
print('Joint mechanism checks passed')
