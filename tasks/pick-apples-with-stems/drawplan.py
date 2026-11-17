"""Plan an axial draw using only ordinary joint-position references.

K^-1 J.T F is a feed-forward position-reference bias for the robot's
unchanged force-drive gains. F is an intended TCP load, not a measured or
directly applied force. Actual seat loads decide physical release elsewhere.
This module has no simulator state, force or effort writer.
"""
from __future__ import annotations

import numpy as np
from yam_arena import ik


def reference(q_up, q_actual, stem_axis, stiffness, limits, effort_limits, velocity_limits,
              base_pos, force_N=22.0, ramp_s=4.0):
    q_up, q_actual = np.asarray(q_up, float), np.asarray(q_actual, float)
    k, bounds = np.asarray(stiffness, float), np.asarray(limits, float)
    efforts, velocities = np.asarray(effort_limits, float), np.asarray(velocity_limits, float)
    u = np.asarray(stem_axis, float)
    arrays = [q_up, q_actual, k, bounds, efforts, velocities, u]
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("Nonfinite axial-reference input")
    if any(a.shape != (6,) for a in [q_up, q_actual, k, efforts, velocities]) or bounds.shape != (6, 2) or u.shape != (3,):
        raise ValueError("Unexpected axial-reference input shape")
    if np.any(k <= 0) or np.any(efforts <= 0) or np.any(velocities <= 0) or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Unknown gains or limits cannot produce a load reference")
    if not np.isfinite([force_N, ramp_s]).all() or not 0 <= force_N <= 22.0 or ramp_s <= 0 or np.linalg.norm(u) < 1e-8:
        raise ValueError("The axial reference is bounded to 0..22 N and a positive ramp")
    low, high = np.maximum(bounds[:, 0], ik.LOWER), np.minimum(bounds[:, 1], ik.UPPER)
    if np.any(q_up < low) or np.any(q_up > high):
        raise ValueError("UP reference is outside the true joint limits")
    jac = np.empty((3, 6))
    eps = 1e-6
    for i in range(6):
        dq = np.eye(6)[i] * eps
        jac[:, i] = (ik.tcp_pose(q_actual + dq, base_pos)[0] - ik.tcp_pose(q_actual - dq, base_pos)[0]) / (2 * eps)
    intended_force = -force_N * u / np.linalg.norm(u)  # apple stem points toward the seat
    extra_torque = jac.T @ intended_force
    increment = extra_torque / k
    target = q_up + increment
    peak_speed = 1.5 * np.abs(increment) / ramp_s  # the standard smoothstep player
    if np.any(target < low) or np.any(target > high):
        raise ValueError("Axial joint reference is unreachable; it must not be silently clipped")
    if np.any(np.abs(extra_torque) > efforts) or np.any(peak_speed > velocities):
        raise ValueError("Axial reference exceeds the unchanged hardware envelope")
    if not np.array_equal(np.clip(target, ik.LOWER, ik.UPPER), target):
        raise ValueError("The standard plan player would clip this reference")
    end_pos, end_rot = ik.tcp_pose(target, base_pos)
    return target, dict(q_UP=q_up.tolist(), q_actual=q_actual.tolist(), stiffness_Nm_per_rad=k.tolist(),
                        limits_rad=bounds.tolist(), effort_limits_Nm=efforts.tolist(), velocity_limits_rad_s=velocities.tolist(),
                        J_TCP_m_per_rad=jac.tolist(), intended_TCP_force_N=intended_force.tolist(),
                        intended_extra_joint_torque_Nm=extra_torque.tolist(), reference_increment_rad=increment.tolist(),
                        joint_target_rad=target.tolist(), peak_increment_speed_rad_s=peak_speed.tolist(),
                        free_reference_endpoint_m=end_pos.tolist(), free_reference_rotation=end_rot.tolist(),
                        ramp_s=float(ramp_s), force_is_planning_reference_not_measurement=True)
