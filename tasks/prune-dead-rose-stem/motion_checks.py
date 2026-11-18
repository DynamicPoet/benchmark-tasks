"""Motion checks for points sampled at a fixed control rate."""

from collections.abc import Sequence

import numpy as np


def finger_pad_points(position, rotation, joint7, joint8, middle, tip):
    """Actual independent YAM finger translations, in the gripper's frame.

    The URDF maps joint7 to +X and joint8 to -X; both closing coordinates
    are nonpositive. Loads can move the fingers asymmetrically.
    """
    position,rotation=np.asarray(position),np.asarray(rotation)
    offsets=(float(joint7),-float(joint8))
    points=np.array([position+rotation@(np.asarray(ref)+[x,0.,0.])
                     for ref in (middle,tip) for x in offsets])
    return points,abs(float(joint7)+float(joint8))


def points_at_rest(history: Sequence[np.ndarray], max_move: float, steps: int) -> bool:
    """Every point must move less than max_move between any two samples in the window."""
    if len(history) <= steps:
        return False
    points = np.asarray(history[-(steps + 1):], dtype=np.float64).reshape(steps + 1, -1, 3)
    displacement = points[:, None] - points[None, :]
    return bool(np.all(np.linalg.norm(displacement, axis=-1) < max_move))


def middle_over_rim(midpoint,position,rotation,height,radius):
    """Vertical projection onto the real circular opening, at its actual pose."""
    midpoint,position,rotation=map(np.asarray,(midpoint,position,rotation))
    normal=rotation[:,2]
    if not all(np.isfinite(x).all()for x in(midpoint,position,rotation))or abs(normal[2])<1e-9:
        return False
    rim=position+rotation@np.array([0.,0.,height])
    crossing=midpoint+np.array([0.,0.,float(normal@(rim-midpoint)/normal[2])])
    local=rotation.T@(crossing-position)
    return bool(np.linalg.norm(local[:2])<=radius)


def ends_over_rim(points,position,rotation,height,radius):
    """Every given point of the cutting, projected straight down, falls inside the opening."""
    return all(middle_over_rim(p,position,rotation,height,radius) for p in points)


def segment_distance(point,a,b):
    """Distance from a point to the segment a-b, in as many dimensions as given."""
    point,a,b=(np.asarray(x,dtype=np.float64) for x in (point,a,b))
    d=b-a
    t=float(np.clip(np.dot(point-a,d)/max(float(np.dot(d,d)),1e-12),0.,1.))
    return float(np.linalg.norm(point-(a+t*d)))


def cutting_held(pads,bottom,top,radius,reach,gap_extra):
    """A gripper holds the cutting when it is closed to at most the cutting's width plus gap_extra and both its finger-pad
    middles are within reach of the cutting's surface. pads maps each arm to (points, gap) from finger_pad_points."""
    held=False
    detail={}
    for side,(points,gap) in pads.items():
        surface=[segment_distance(p,bottom,top)-radius for p in points[:2]]
        holds=bool(gap<=2*radius+gap_extra and all(d<=reach for d in surface))
        detail[side]={'finger_gap_m':float(gap),'pad_middle_to_surface_m':surface,'holds':holds}
        held=held or holds
    return held,detail


def clear_of_tub(points,position,rotation,height,radius):
    """No point is over the tub seen from above: its axis from base to rim, widened by its rim radius (a tipped tub too)."""
    position,rotation=np.asarray(position,dtype=np.float64),np.asarray(rotation,dtype=np.float64)
    a=position[:2]
    b=(position+rotation@np.array([0.,0.,height]))[:2]
    return all(segment_distance(np.asarray(p)[:2],a,b)>radius for p in points)
