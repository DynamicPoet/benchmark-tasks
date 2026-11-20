"""poseplay: two more kinds of waypoint for ``yam_scripted``'s plan player, so a demo can hold the gripper at any angle and move in joint space.

``yam_scripted`` plays ``demo()`` as straight-down Cartesian waypoints (``yam_arena.demo.Waypoint``). Importing this module adds:

- :class:`Pose`: a Cartesian waypoint with any gripper orientation (``rot``, a 3x3 rotation of the tool frame in the world).
- :class:`Joints`: a waypoint in joint space (``q``, the six arm joint targets), interpolated smoothly, or held from the
  first control step with ``step=True`` (a swing: the target jumps, and the arm's own gains and torque limits set the speed).

Any waypoint's ``pos`` (a :class:`Pose`'s ``rot``, a :class:`Joints` waypoint's ``q``) may instead be a callable, called
once with the plan when the waypoint starts; it may read the simulator through ``scenekit.env()`` (feedback from where
things actually are), and must return a nominal value when there is no simulator (offline plan checks).

It patches ``ArmPlan.action7`` in place; plain waypoints play exactly as before. Pure Python and NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from yam_arena import ik
from yam_arena.demo import ArmPlan, Waypoint, grip_rotation

__all__ = ["Joints", "Pose"]


@dataclass
class Pose(Waypoint):
    """A Cartesian waypoint with the tool frame turned to ``rot`` (3x3, rows of the world matrix)."""

    rot: tuple = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


@dataclass
class Joints(Waypoint):
    """A joint-space waypoint: ``q`` are the six arm joint targets (rad). ``step`` sends them at once instead of easing."""

    q: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    step: bool = False


def _value(x, plan):
    return x(plan) if callable(x) else x


def _rot(w: Waypoint, plan) -> np.ndarray:
    return np.asarray(_value(w.rot, plan), float) if isinstance(w, Pose) else grip_rotation(w.yaw)


def _begin(plan: ArmPlan, k: int) -> None:
    """Enter waypoint k: resolve its targets and record where it starts from (the previous waypoint's end)."""
    w = plan.waypoints[k]
    st = plan._hp
    if k > 0:
        prev = st["seg"]
        if prev["kind"] == "joints":
            st["q"] = prev["q1"].copy()
            st["pos"], st["rot"] = ik.tcp_pose(st["q"], plan.base_pos)
        else:
            st["pos"], st["rot"] = prev["pos1"], prev["rot1"]
            q, _, _ = ik.solve(prev["pos1"], ik.matrix_to_quat_xyzw(prev["rot1"]), plan.q, base_pos=plan.base_pos, iters=80)
            st["q"] = q
        st["finger"] = prev["finger1"]
    seg = {"finger0": st["finger"], "finger1": float(w.finger), "label": w.label}
    if isinstance(w, Joints):
        seg.update(kind="joints", q0=st["q"].copy(), q1=np.clip(np.asarray(_value(w.q, plan), float), ik.LOWER, ik.UPPER),
                   step=bool(w.step))
    else:
        rot1 = _rot(w, plan)
        seg.update(kind="cart", pos0=np.asarray(st["pos"], float), rot0=st["rot"], pos1=np.asarray(_value(w.pos, plan), float),
                   rot1=rot1)
    st["seg"] = seg


def action7(self: ArmPlan, t: float):
    """[j1..j6, normalized grip] at time t, and the waypoint's label. Called once per control step, t increasing."""
    if not hasattr(self, "_hp"):
        self._hp = {"k": -1, "q": self.q.copy(), "pos": self.start_pos, "rot": self.start_rot, "finger": self.finger_start}
    st = self._hp
    acc, k = 0.0, 0
    for k, w in enumerate(self.waypoints):
        if t < acc + w.duration:
            break
        acc += w.duration
    else:
        k, acc = len(self.waypoints) - 1, acc - self.waypoints[-1].duration
    while st["k"] < k:
        st["k"] += 1
        _begin(self, st["k"])
    seg, w = st["seg"], self.waypoints[k]
    s = ik.smoothstep(min(1.0, max(0.0, (t - acc) / w.duration)))
    finger = seg["finger0"] + s * (seg["finger1"] - seg["finger0"])
    if seg["kind"] == "joints":
        q = seg["q1"] if seg["step"] else seg["q0"] + s * (seg["q1"] - seg["q0"])
        self.q = q.copy()
    else:
        pos = seg["pos0"] + s * (seg["pos1"] - seg["pos0"])
        rot = ik.slerp_matrix(seg["rot0"], seg["rot1"], s)
        q, pe, re = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), self.q, base_pos=self.base_pos, iters=60)
        self.q = q
        self.worst_ik_mm = max(self.worst_ik_mm, pe * 1000)
        self.worst_ik_deg = max(self.worst_ik_deg, float(np.degrees(re)))
    grip = float(np.clip(finger / (-0.04695), 0.0, 1.0))
    label = seg["label"] if t < acc + w.duration or k < len(self.waypoints) - 1 else "done"
    return np.r_[self.q, grip], label


ArmPlan.action7 = action7
