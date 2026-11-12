"""Finite blade footprints and convex plant cross-sections, in the cutting plane."""

import numpy as np


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def footprint_distance(point, vertices, faces):
    """Distance to the union of projected triangles, including their interiors."""
    triangles = vertices[faces, :2]
    edges = np.roll(triangles, -1, axis=1) - triangles
    delta = np.asarray(point) - triangles
    sides = cross(edges, delta)
    area = cross(edges[:, 0], triangles[:, 2] - triangles[:, 0])
    inside = ((sides >= -1e-12).all(axis=1) | (sides <= 1e-12).all(axis=1)) & (abs(area) > 1e-12)
    if inside.any():
        return 0.0
    fraction = np.clip((delta * edges).sum(axis=2) / np.maximum((edges * edges).sum(axis=2), 1e-24), 0, 1)
    return float(np.linalg.norm(delta - fraction[..., None] * edges, axis=2).min())


def convex_section(points):
    """Convex hull of a plane section of the convex bud mesh, counterclockwise."""
    points = sorted(set(tuple(map(float, p)) for p in points))
    if len(points) < 3:
        return np.empty((0, 2))
    def half(sequence):
        out = []
        for point in sequence:
            while len(out) >= 2 and cross(np.subtract(out[-1], out[-2]), np.subtract(point, out[-1])) <= 1e-14:
                out.pop()
            out.append(point)
        return out
    return np.asarray(half(points)[:-1] + half(points[::-1])[:-1])


def _clip(subject, a, b):
    out = []
    for start, end in zip(subject, np.roll(subject, -1, axis=0)):
        ds, de = cross(b - a, start - a), cross(b - a, end - a)
        if ds >= -1e-12:
            out.append(start)
        if (ds >= 0) != (de >= 0):
            out.append(start + (end - start) * ds / (ds - de))
    return np.asarray(out).reshape(-1, 2)


def section_contact(points, vertices, faces, reach):
    """Return an actual blade/section overlap point in the finite jaw reach.

    Triangle clipping covers crossing edges and either region containing the other;
    checking only section vertices would miss a narrow blade through its interior.
    """
    section = convex_section(points)
    if len(section) < 3:
        return None
    for triangle in vertices[faces, :2]:
        polygon = triangle
        for a, b in zip(section, np.roll(section, -1, axis=0)):
            polygon = _clip(polygon, a, b)
            if len(polygon) < 3:
                break
        if len(polygon) < 3 or abs(cross(polygon, np.roll(polygon, -1, axis=0)).sum()) < 1e-14:
            continue
        edges = np.roll(polygon, -1, axis=0) - polygon
        fraction = np.clip(-(polygon * edges).sum(axis=1) / np.maximum((edges * edges).sum(axis=1), 1e-24), 0, 1)
        closest = polygon + fraction[:, None] * edges
        low = closest[np.linalg.norm(closest, axis=1).argmin()]
        if footprint_distance(np.zeros(2), polygon, np.array([[0, i, i + 1] for i in range(1, len(polygon) - 1)])) == 0:
            low = np.zeros(2)
        high = polygon[np.linalg.norm(polygon, axis=1).argmax()]
        if np.linalg.norm(low) > reach[1] or np.linalg.norm(high) < reach[0]:
            continue
        if np.linalg.norm(low) >= reach[0]:
            return low
        # A continuous segment from the closest point to the farthest vertex
        # crosses the inner reach circle inside this convex overlap polygon.
        lo, hi = 0.0, 1.0
        for _ in range(40):
            t = (lo + hi) / 2
            if np.linalg.norm(low + t * (high - low)) < reach[0]:
                lo = t
            else:
                hi = t
        return low + hi * (high - low)
    return None
