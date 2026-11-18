"""Task-local full-pose Cartesian waypoints, emitting ordinary YAM joint actions."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from yam_arena import ik
from yam_arena.demo import ArmPlan, Waypoint, grip_rotation


@dataclass
class PoseWaypoint(Waypoint):
    rotation: np.ndarray | None = None
    nominal_jaw: float = 0.0  # Offline geometry reference only; never written to the object.
    joint_start: np.ndarray | None = None
    joint_goal: np.ndarray | None = None


def closed_hand_pose(capture,label,clock,boundaries,nominal_pos,nominal_R):
    """Ordinary closed withdrawal: preserve the reachable wrist until clear."""
    alpha=0.
    if label=='to a free spot':
        begin,end=boundaries[('right',label)]
        alpha=ik.smoothstep((clock-begin)/(end-begin))
    offset=np.asarray(capture['TCP'])-np.asarray(capture['nominal_TCP'])
    p=np.asarray(nominal_pos)+(1-alpha)*offset
    R=ik.slerp_matrix(np.asarray(capture['R']),np.asarray(nominal_R),alpha)
    return p,R


class PoseArmPlan(ArmPlan):
    def target(self, t):
        pos,rot,finger=self.start_pos,self.start_rot,self.finger_start
        for w in self.waypoints:
            goal=np.asarray(w.pos,float)
            r=grip_rotation(w.yaw) if getattr(w,'rotation',None) is None else np.asarray(w.rotation,float)
            if t<w.duration:
                a=ik.smoothstep(t/w.duration)
                if getattr(w,'joint_goal',None) is not None:
                    q=np.asarray(w.joint_start)+a*(np.asarray(w.joint_goal)-w.joint_start)
                    p,r=ik.tcp_pose(q,self.base_pos)
                    return p,r,finger+a*(w.finger-finger),w.label
                return pos+a*(goal-pos),ik.slerp_matrix(rot,r,a),finger+a*(w.finger-finger),w.label
            t-=w.duration
            pos,rot,finger=goal,r,w.finger
        return pos,rot,finger,'done'

    def action7(self,t):
        remaining=t
        start_finger=self.finger_start
        for w in self.waypoints:
            if remaining<w.duration:
                if getattr(w,'joint_goal',None) is not None:
                    a=ik.smoothstep(remaining/w.duration)
                    self.q=np.asarray(w.joint_start)+a*(np.asarray(w.joint_goal)-w.joint_start)
                    finger=start_finger+a*(w.finger-start_finger)
                    from yam_arena.constants import FINGER_OPEN
                    return np.r_[self.q,np.clip(finger/FINGER_OPEN,0.,1.)],w.label
                break
            remaining-=w.duration
            start_finger=w.finger
        return super().action7(t)

    def jaw_reference(self,t):
        """Assumed symmetric opening for CPU clearance, not an observation or command."""
        q=0.
        for w in self.waypoints:
            goal=getattr(w,'nominal_jaw',q)
            if t<w.duration:
                return q+ik.smoothstep(t/w.duration)*(goal-q)
            t-=w.duration;q=goal
        return q


def replace_task_plans(policy, task_path):
    task=getattr(policy,'_task',None)
    if task is None or Path(task.path).resolve()!=Path(task_path).resolve():
        return
    for arm,plan in list(policy._plans.items()):
        if any(isinstance(w,PoseWaypoint) for w in plan.waypoints):
            policy._plans[arm]=PoseArmPlan(plan.waypoints,policy._bases[arm])


def install(task_path,controller_factory=None):
    """Adapt only this task's scripted plan; all emitted commands use the stock IK action API."""
    from yam_arena.scripted import YamScriptedPolicy
    if getattr(YamScriptedPolicy,'_pruning_pose_plans',False):
        return
    original=YamScriptedPolicy.reset
    original_act=YamScriptedPolicy.act
    def reset(self,scene):
        original(self,scene)
        self._pruning_controller=None
        replace_task_plans(self,task_path)
        task=getattr(self,'_task',None)
        if controller_factory is not None and task is not None and Path(task.path).resolve()==Path(task_path).resolve():
            self._pruning_controller=controller_factory(self)
    def act(self,observation):
        controller=getattr(self,'_pruning_controller',None)
        return controller.act(observation) if controller is not None else original_act(self,observation)
    YamScriptedPolicy.reset=reset
    YamScriptedPolicy.act=act
    YamScriptedPolicy._pruning_pose_plans=True
