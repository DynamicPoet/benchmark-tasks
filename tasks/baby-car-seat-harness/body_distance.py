"""Conservative signed Euclidean distance bounds for an ellipsoid union.

Closest points on each ellipsoid use the scalar normal-equation root. The
outside-union distance is exact up to numerical tolerance. Within overlapping
solids we bound the distance to the external union boundary from both sides;
we never mistake an internal ellipsoid face for the doll's outer surface.
"""

from __future__ import annotations

import numpy as np


def ellipsoid_closest(points, centre, axes):
    p = np.asarray(points, float) - np.asarray(centre, float)
    r = np.asarray(axes, float)
    if np.ptp(r) < 1e-14:
        norm = np.linalg.norm(p, axis=1)
        closest = p * (r[0] / np.maximum(norm, 1e-30))[:, None]
        closest[norm == 0] = (r[0], 0.0, 0.0)
        return closest + centre, np.abs(norm - r[0]), norm < r[0]
    square = r * r
    level = np.sum((p / r) ** 2, axis=1)
    inside = level < 1.0
    smallest = int(np.argmin(r))
    low = np.where(inside, -square[smallest] * (1 - 1e-12), 0.0)
    high = np.where(inside, 0.0, np.linalg.norm(p, axis=1) * float(np.max(r)))

    def f(value):
        return np.sum(square * p * p / (square + value[:, None]) ** 2, axis=1) - 1

    singular = inside & (f(low) <= 0)
    for _ in range(48):
        middle = (low + high) / 2
        above = f(middle) > 0
        low = np.where(above, middle, low)
        high = np.where(above, high, middle)
    closest = square * p / (square + ((low + high) / 2)[:, None])
    if np.any(singular):
        other = [axis for axis in range(3) if axis != smallest]
        candidate = np.zeros((int(singular.sum()), 3))
        candidate[:, other] = square[other] * p[singular][:, other] / (square[other] - square[smallest])
        remainder = np.maximum(0.0, 1 - np.sum((candidate[:, other] / r[other]) ** 2, axis=1))
        candidate[:, smallest] = r[smallest] * np.sqrt(remainder) * np.where(p[singular, smallest] < 0, -1, 1)
        closest[singular] = candidate
    on = np.abs(level - 1.0) <= 1e-13
    closest[on] = p[on]
    distance = np.linalg.norm(closest - p, axis=1)
    return closest + centre, distance, inside


def union_distance_bounds(points, bodies):
    """Return lower/upper bounds on signed distance: negative means inside."""
    p = np.asarray(points, float)
    distances, interiors, candidates = [], [], []
    for centre, axes in bodies:
        closest, distance, inside = ellipsoid_closest(p, centre, axes)
        candidates.append(closest)
        distances.append(distance)
        interiors.append(inside)
    distances, interiors = np.asarray(distances), np.asarray(interiors)
    inside = np.any(interiors, axis=0)
    outside_distance = np.min(distances, axis=0)
    lower = outside_distance.copy()
    upper = outside_distance.copy()
    if np.any(inside):
        # Exiting the union requires exiting every individual solid containing
        # the point. This gives a lower bound on penetration depth.
        depth_lower = np.max(np.where(interiors, distances, 0.0), axis=0)
        # The front-most intersection on the +w ray is certainly on the outer
        # boundary; its distance provides a conservative exit-distance bound.
        front = np.full(len(p), -np.inf)
        for centre, axes in bodies:
            radial = ((p[:, 0] - centre[0]) / axes[0]) ** 2 + ((p[:, 1] - centre[1]) / axes[1]) ** 2
            height = centre[2] + axes[2] * np.sqrt(np.maximum(0.0, 1 - radial))
            front = np.maximum(front, np.where(radial <= 1, height, -np.inf))
        depth_upper = np.maximum(0.0, front - p[:, 2])
        for closest, distance in zip(candidates, distances):
            external = np.ones(len(p), bool)
            for centre, axes in bodies:
                external &= np.sum(((closest - centre) / axes) ** 2, axis=1) >= 1 - 1e-9
            depth_upper = np.minimum(depth_upper, np.where(external, distance, np.inf))
        lower[inside] = -depth_upper[inside]
        upper[inside] = -depth_lower[inside]
    return lower - 1e-9, upper + 1e-9


def certify_surface(triangles, bodies, penetration=0.002, gap=0.004, max_depth=6):
    """Certify every point of each actual triangle, with fail-closed subdivision.

    True signed distance is 1-Lipschitz. At a triangle centroid, its distance
    interval expanded by the farthest-vertex radius bounds the entire face.
    Uncertain faces are split into four; unresolved faces never pass.
    """
    todo = np.asarray(triangles, float)
    certified_low, certified_high = [], []
    for depth in range(max_depth + 1):
        if not len(todo):
            return True, float(min(certified_low)), float(max(certified_high))
        centres = todo.mean(axis=1)
        radius = np.max(np.linalg.norm(todo - centres[:, None], axis=2), axis=1)
        low, high = union_distance_bounds(centres, bodies)
        whole_low, whole_high = low - radius, high + radius
        valid = (whole_low >= -penetration) & (whole_high <= gap)
        if np.any(valid):
            certified_low.append(float(whole_low[valid].min()))
            certified_high.append(float(whole_high[valid].max()))
        if np.any((high < -penetration) | (low > gap)):
            return False, float(whole_low.min()), float(whole_high.max())
        pending = todo[~valid]
        if not len(pending):
            return True, float(min(certified_low)), float(max(certified_high))
        if depth == max_depth:
            return False, float(whole_low.min()), float(whole_high.max())
        a, b, c = pending[:, 0], pending[:, 1], pending[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        todo = np.concatenate([np.stack(t, axis=1) for t in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])
    raise AssertionError("Unreachable")
