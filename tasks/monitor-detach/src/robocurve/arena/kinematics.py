"""Forward and inverse kinematics of both YAM arms on yam_arena's URDF (NumPy only).

Same frames and method as the policies were tuned with: the station frame (origin at the left arm base,
right base at y = -0.61) and damped least squares on the tool point and the flange rotation. The tool point
is 0.14465 m out along the flange's +z, between the pads' grip centres; yam_arena's ``TCP_OFFSET`` is 10.65 mm
shorter. Pure Python at import time apart from yam_arena's URDF reader, so it runs in the yam_arena environment.
"""

from __future__ import annotations

import numpy as np

from robocurve.sim.arm_math import ARM_JOINTS, IkResult, Side, Vec, rotation_error

TCP_ALONG_FLANGE_Z = 0.14465  # m
BASES: dict[str, Vec] = {"left": np.zeros(3), "right": np.array([0.0, -0.61, 0.0])}


class ArenaKinematics:
    """Kinematics of both arms from yam_arena's URDF, with the interface the scripted policies use."""

    def __init__(self) -> None:
        from yam_arena import YAM_URDF
        from yam_arena.urdf import forward_kinematics, parse_joints

        self._urdf, self._fk = YAM_URDF, forward_kinematics
        joints = parse_joints(YAM_URDF)
        self._joints = [joints[name] for name in ARM_JOINTS]
        self._range = np.array([[j.lower, j.upper] for j in self._joints])

    def joint_range(self, side: Side) -> Vec:
        """(6, 2) lower/upper limits of the arm joints [rad]."""
        return self._range.copy()

    def _pose_and_jacobian(self, side: Side, q: Vec) -> tuple[Vec, Vec, Vec]:
        links = self._fk(self._urdf, {j.name: float(v) for j, v in zip(self._joints, q, strict=True)})
        flange = links["gripper"]
        pos = flange[:3, 3] + flange[:3, 2] * TCP_ALONG_FLANGE_Z
        jac = np.zeros((6, 6))
        for i, joint in enumerate(self._joints):
            frame = links[joint.child]  # the joint's axis is fixed in its child link
            axis = frame[:3, :3] @ joint.axis
            jac[:3, i] = np.cross(axis, pos - frame[:3, 3])
            jac[3:, i] = axis
        return pos + BASES[side], flange[:3, :3].copy(), jac

    def fk(self, side: Side, q: Vec) -> tuple[Vec, Vec]:
        """TCP position and flange rotation for arm joint angles ``q``."""
        pos, rot, _ = self._pose_and_jacobian(side, q)
        return pos, rot

    def static_torques(self, side: Side, q: Vec, force: Vec) -> Vec:
        """Arm joint torques that balance a world-frame force applied at the TCP (tau = J^T F)."""
        _, _, jac = self._pose_and_jacobian(side, q)
        return jac[:3].T @ np.asarray(force, dtype=float)

    def solve(
        self,
        side: Side,
        pos: Vec,
        rot: Vec,
        q_init: Vec | None = None,
        iters: int = 300,
        damping: float = 1e-2,
        rot_weight: float = 0.3,
    ) -> IkResult:
        """Damped least-squares IK for the TCP position and flange rotation, within joint limits."""
        lo, hi = self._range[:, 0], self._range[:, 1]
        q = np.clip(np.zeros(6) if q_init is None else np.asarray(q_init, dtype=float), lo, hi)
        weights = np.array([1.0] * 3 + [rot_weight] * 3)
        for _ in range(iters):
            cur_pos, cur_rot, jac = self._pose_and_jacobian(side, q)
            err = np.concatenate([pos - cur_pos, rotation_error(rot, cur_rot)])
            if np.linalg.norm(err[:3]) < 1e-4 and np.linalg.norm(err[3:]) < 1e-3:
                break
            weighted = jac * weights[:, None]
            dq = weighted.T @ np.linalg.solve(weighted @ weighted.T + damping**2 * np.eye(6), err * weights)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), lo, hi)
        cur_pos, cur_rot = self.fk(side, q)
        return IkResult(q=q, pos_err=float(np.linalg.norm(pos - cur_pos)),
                        rot_err=float(np.linalg.norm(rotation_error(rot, cur_rot))))

    def solve_best(
        self, side: Side, pos: Vec, rot: Vec, seeds: list[Vec] | None = None, restarts: int = 12, seed: int = 0
    ) -> IkResult:
        """Run :meth:`solve` from the given seeds plus random restarts and keep the best result."""
        rng = np.random.default_rng(seed)
        starts = list(seeds or []) + [rng.uniform(self._range[:, 0], self._range[:, 1]) for _ in range(restarts)]
        best: IkResult | None = None
        for start in starts:
            result = self.solve(side, pos, rot, q_init=start)
            if best is None or (result.pos_err + 0.05 * result.rot_err) < (best.pos_err + 0.05 * best.rot_err):
                best = result
            if best.ok and best.pos_err < 5e-4:
                break
        assert best is not None
        return best
