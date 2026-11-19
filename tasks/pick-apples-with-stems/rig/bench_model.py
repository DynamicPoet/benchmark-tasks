"""Model predictions for the assembled rig under a pull through the fruit centre.

Run with the task's Python environment. Forces are additional applied tensions
in the gauge line. Positive angles point below the initial outward stem axis.
This fixed-tree numerical experiment has no gripper or physical measurements.
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np


def advance_rotation(rotation, angular_step, skew):
    angle = np.linalg.norm(angular_step)
    if angle < 1e-12:
        return rotation
    axis = skew(angular_step / angle)
    increment = np.eye(3) + math.sin(angle) * axis + (1 - math.cos(angle)) * (axis @ axis)
    return increment @ rotation


def simulate_pull(assets, joint_model, station, angle_deg, ramp, hz):
    centre, rotation = assets.apple_rest(station)
    anchor = assets.ABSCISSION.copy()
    mass = assets.APPLE_MASS
    inertia = 0.4 * mass * (assets.APPLE_D / 2) ** 2
    joint = joint_model.Joint(station, anchor, assets.abscission_in_tree(station),
                              rotation.copy(), mass, inertia)
    zero = np.zeros(3)
    down = np.array([0., 0., -1.])
    gravity = 9.81 * down
    identity = np.eye(3)
    joint.settle(gravity, identity)
    outward = -rotation[:, 0]
    angle = math.radians(angle_deg)
    direction = math.cos(angle) * outward + math.sin(angle) * down
    velocity = zero.copy()
    angular_velocity = zero.copy()
    dt = 1.0 / hz
    gauge_force = 0.
    # The previous integration step applied gauge_force, so pair that value
    # with the resulting internal load and first separation event.
    for step in range(hz + math.ceil(45 / ramp * hz) + 2):
        force, moment, _, loads = joint.wrench(
            centre, rotation, velocity, angular_velocity, zero, identity, zero, zero, zero, dt)
        site = joint.check(loads)
        if site is not None:
            return site, gauge_force, loads['tension'], loads['bending']
        gauge_force = min(45., max(0., step * dt - 1.) * ramp)
        velocity += (force + gauge_force * direction + mass * gravity) * (dt / mass)
        angular_velocity += (moment + np.cross(rotation @ anchor, force)) * (dt / inertia)
        centre = centre + velocity * dt
        rotation = advance_rotation(rotation, angular_velocity * dt, joint_model.skew)
    return 'no_release', gauge_force, loads['tension'], loads['bending']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-dir', type=Path, default=Path(__file__).resolve().parents[1],
                        help='directory containing scene_assets.py and stemjoint.py')
    parser.add_argument('--hz', type=int, default=240, choices=(240, 480))
    parser.add_argument('--ramp', type=float, default=4.0, help='force ramp in N/s')
    args = parser.parse_args()
    if not math.isfinite(args.ramp) or args.ramp <= 0:
        parser.error('the ramp must be finite and positive')
    sys.path.insert(0, str(args.task_dir.resolve()))
    import scene_assets
    import stemjoint
    writer = csv.writer(sys.stdout, lineterminator='\n')
    writer.writerow(['source','station','angle_below_outward_deg','ramp_N_per_s','step_Hz',
                     'first_release_site','applied_gauge_N','model_stem_tension_N','model_bending_Nm'])
    for station in scene_assets.SPURS:
        for angle in (0., -15., 15., 90.):
            site, gauge, tension, bending = simulate_pull(scene_assets, stemjoint, station, angle,
                                                         args.ramp, args.hz)
            writer.writerow(['model',station,angle,args.ramp,args.hz,site,f'{gauge:.3f}',
                             f'{tension:.3f}',f'{bending:.5f}'])


if __name__ == '__main__':
    main()
