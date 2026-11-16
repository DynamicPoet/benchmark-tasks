"""Inspect Robots policy ``wheelchair_scripted``: plays the task file's ``demo()`` like ``yam_scripted``, with feedback.

The YAM's wrist joints are soft (stiffness 10 N m/rad), so a gripper pushing sideways on a load lags its command by
about 5 mm per newton, and pressing down by about 2.5 mm per newton. Opening the wheelchair takes tens of newtons at
each seat rail, so an open-loop pair of arms lets whichever side of the chair breaks loose first run ahead, and the
chair slides towards it. During waypoints whose label starts with ``track`` the two arms are therefore coordinated from
their own measured joint angles (forward kinematics gives where each gripper is), with integral action:

- the two grippers' common motion: down (z) they press until both are as low as the plan says; sideways (y) and
  along the chair (x) it is held where the plan puts it, so if the chair starts to slide or roll one way, both arms
  push it back together;
- their difference: in height (z) and along the chair (x) it is held at zero, so one side cannot sink or swing ahead
  of the other and the chair does not turn; their opening (y) follows the plan only weakly, so neither arm drags its
  side out ahead of the press.

During waypoints labelled ``ease`` the corrections fade out, so the arms stop pushing before they let go. Everything
else is played as ``yam_scripted`` plays it: smooth Cartesian waypoints, IK on the URDF, open loop. The policy reads
only the robot's own joint positions (the observation's ``joint_pos``) and the clock.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from inspect_robots.policy import PolicyBase, PolicyInfo
from inspect_robots.types import Action, ActionChunk, Observation
from inspect_robots_yam.config import action_box
from inspect_robots_yam.packing import STATE_KEY, pack, split

from yam_arena import ik, station, task_file
from yam_arena.constants import CONTROL_HZ, FINGER_OPEN, READY
from yam_arena.demo import ArmPlan

ARMS = ("left", "right")


class WheelchairScriptedPolicy(PolicyBase):
    # Per axis (x, y, z): proportional gain, integral gain (1/s) and limit (m), for the grippers' common motion and
    # for their difference (left minus right, halved).
    COMMON = dict(kp=(0.5, 1.0, 0.0), ki=(2.0, 8.0, 4.0), limit=(0.10, 0.15, 0.16))
    DIFF = dict(kp=(1.0, 0.0, 1.0), ki=(6.0, 1.0, 4.0), limit=(0.12, 0.06, 0.08))

    def __init__(self, hold_s: float = 3.0, ease_s: float = 0.6) -> None:
        low, high = np.append(ik.LOWER, 0.0), np.append(ik.UPPER, 1.0)
        self.info = PolicyInfo(name="wheelchair_scripted", action_space=action_box(pack(low, low), pack(high, high)))
        self.hold_s, self.ease_s = float(hold_s), float(ease_s)
        self._dt = 1.0 / CONTROL_HZ
        self._task = None
        self._bases = dict(station.BASES)
        self._plans: dict[str, ArmPlan] = {}
        self._clear()
        self._end = 0.0

    def _clear(self) -> None:
        self._int = {"common": np.zeros(3), "diff": np.zeros(3)}
        self._prop = {"common": np.zeros(3), "diff": np.zeros(3)}
        self._single = {arm: np.zeros(3) for arm in ARMS}

    def bind(self, embodiment_info: Any) -> None:
        if embodiment_info.control_hz:
            self._dt = 1.0 / float(embodiment_info.control_hz)
        self._task = task_file.lookup(embodiment_info.environment_id)
        if self._task is None or self._task.demo is None:
            raise ValueError("wheelchair_scripted plays a task file's demo(): run it with -E task=task.py")

    def reset(self, scene: Any) -> None:
        plan = dict(self._task.demo())
        self._plans = {arm: ArmPlan(list(w), self._bases[arm]) for arm, w in plan.items()}
        self._clear()
        self._end = max((p.duration for p in self._plans.values()), default=0.0)

    def _update(self, now: float, measured: dict, labels: dict) -> None:
        """Integrate the corrections from where the grippers are against where the plan wants them now."""
        tracked = [a for a in ARMS if a in self._plans and labels[a].startswith("track")]
        easing = [a for a in ARMS if a in self._plans and labels[a].startswith("ease")]
        if not tracked:
            fade = np.exp(-self._dt / self.ease_s) if easing else 0.0
            for d in (self._int, self._prop, self._single):
                for k in d:
                    d[k] = d[k] * fade
            return
        err = {}
        for a in tracked:
            plan = self._plans[a]
            err[a] = plan.target(now)[0] - ik.tcp_pose(measured[a][:6], plan.base_pos)[0]
        if len(tracked) == 1:  # one arm alone presses down only
            a = tracked[0]
            self._single[a][2] = float(np.clip(self._single[a][2] + 4.0 * err[a][2] * self._dt, -0.16, 0.16))
            return
        parts = {"common": (err["left"] + err["right"]) / 2, "diff": (err["left"] - err["right"]) / 2}
        for k, gains in (("common", self.COMMON), ("diff", self.DIFF)):
            ki, kp, lim = (np.array(gains[n]) for n in ("ki", "kp", "limit"))
            self._int[k] = np.clip(self._int[k] + ki * parts[k] * self._dt, -lim, lim)
            self._prop[k] = kp * parts[k]

    def _offset(self, arm: str) -> np.ndarray:
        side = 1.0 if arm == "left" else -1.0
        common = np.clip(self._int["common"] + self._prop["common"], -0.18, 0.18)
        diff = np.clip(self._int["diff"] + self._prop["diff"], -0.15, 0.15)
        return common + side * diff + self._single[arm]

    def act(self, observation: Observation) -> ActionChunk:
        now = float(observation.state_time)
        t = now + self._dt  # where the arm should be at the end of this control step
        measured = dict(zip(ARMS, split(np.asarray(observation.state[STATE_KEY], dtype=float)), strict=True))
        targets = {a: p.target(t) for a, p in self._plans.items()}
        self._update(now, measured, {a: targets[a][3] if a in targets else "" for a in ARMS})
        per_arm = []
        for arm in ARMS:
            plan = self._plans.get(arm)
            if plan is None:
                per_arm.append(np.r_[READY, 1.0])
                continue
            pos, rot, finger, _ = targets[arm]
            off = self._offset(arm)
            q, _, _ = ik.solve(pos + off, ik.matrix_to_quat_xyzw(rot), plan.q, base_pos=plan.base_pos, iters=60)
            plan.q = q
            per_arm.append(np.r_[q, float(np.clip(finger / FINGER_OPEN, 0.0, 1.0))])
        log = os.environ.get("TASK_SCORE_DIR")
        if log:  # the corrections, for checking the force budget afterwards
            os.makedirs(log, exist_ok=True)
            with open(os.path.join(log, "policy.jsonl"), "a") as f:
                f.write(json.dumps({"t": round(now, 3), **{a: [round(float(v), 4) for v in self._offset(a)]
                                                          for a in ARMS}}) + "\n")
        meta = {"request_stop": True} if t > self._end + self.hold_s else {}
        return ActionChunk(actions=[Action(data=pack(*per_arm), meta=meta)], control_hz=1.0 / self._dt)


def wheelchair_scripted(**kwargs: Any) -> WheelchairScriptedPolicy:
    """Policy ``wheelchair_scripted``: the task's demo() with the arms coordinated on ``track`` waypoints."""
    return WheelchairScriptedPolicy(**kwargs)
