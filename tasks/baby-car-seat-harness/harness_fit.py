"""Geometric fitting evidence for a practice-doll harness.

Only the fixed-endpoint shoulder assay has a material/chord certificate.
It does not model material entering from outside that window, human fingers,
cloth friction, or a child restraint in a crash.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from body_distance import certify_surface
from pinch_assay import assess_patch

STRAP_WIDTH = 0.022
WIDTH_SAMPLES = 11
CENTRE_COLUMN = WIDTH_SAMPLES // 2
DENSE_STEPS = (4, 4, 4, 16, 20, 8)
CONTROL_ROWS = tuple(np.r_[0, np.cumsum(DENSE_STEPS)].astype(int))
LONGITUDINAL_SAMPLES = int(CONTROL_ROWS[-1] + 1)
CONTACT_SLICE = slice(CONTROL_ROWS[1], CONTROL_ROWS[5] + 1)
SHOULDER_SLICE = slice(CONTROL_ROWS[1], CONTROL_ROWS[3] + 1)
SHOULDER_SLACK = 0.025
FOLD_OVERLAP = 0.002
PENETRATION = 0.002
SURFACE_GAP = 0.004
MATERIAL_TOLERANCE = 0.0005
OVERDRAW_TOLERANCE = 0.001
CONTACT_ROWS = (1, 2, 3, 4, 5)
BODY = (
    ((0.072, 0.0, 0.018), (0.090, 0.062, 0.028)),
    ((0.135, 0.058, 0.024), (0.022, 0.020, 0.018)),
    ((0.135, -0.058, 0.024), (0.022, 0.020, 0.018)),
    ((0.156, 0.0, 0.026), (0.022, 0.030, 0.020)),
)


def surface_w(u, v):
    """Front of the modeled doll, measured along the backrest-normal w axis."""
    u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
    result = np.full(u.shape, np.nan)
    for centre, semi in BODY:
        radial = ((u - centre[0]) / semi[0]) ** 2 + ((v - centre[1]) / semi[1]) ** 2
        candidate = np.where(radial <= 1, centre[2] + semi[2] * np.sqrt(np.maximum(0.0, 1 - radial)), np.nan)
        result = np.fmax(result, candidate)
    return result


def fitted_centres(side: int, tongue: np.ndarray) -> np.ndarray:
    """Reference is determined by the doll, shoulder exit and actual buckle anchor."""
    u = np.array([0.155, 0.140, 0.130, 0.120, 0.075, 0.020, float(tongue[0])])
    v = np.array([side * 0.035] * 4 + [side * 0.029, side * 0.022, float(tongue[1])])
    w = surface_w(u, v) + 0.001
    w[0], w[-1] = 0.055, float(tongue[2])
    return np.column_stack([u, v, w])


def dense_centres(centres: np.ndarray) -> np.ndarray:
    """The displayed centreline follows body curvature between contact controls."""
    control = np.asarray(centres, float)
    rows = np.arange(LONGITUDINAL_SAMPLES)
    interval = np.minimum(np.searchsorted(CONTROL_ROWS, rows, side="right") - 1, 5)
    t = (rows - np.asarray(CONTROL_ROWS)[interval]) / np.asarray(DENSE_STEPS)[interval]
    a, b = control[interval], control[interval + 1]
    xyz = a + t[:, None] * (b - a)
    control_surface = surface_w(control[:, 0], control[:, 1])
    base = surface_w(xyz[:, 0], xyz[:, 1])
    surf_a, surf_b = control_surface[interval], control_surface[interval + 1]
    valid = (interval >= 1) & (interval <= 4) & np.isfinite(surf_a) & np.isfinite(surf_b) & np.isfinite(base)
    gap = (1 - t) * (a[:, 2] - surf_a) + t * (b[:, 2] - surf_b)
    xyz[valid, 2] = base[valid] + gap[valid]
    return xyz


def reference_length(centres: np.ndarray) -> float:
    return polyline_length(dense_centres(centres))


def ribbon_points(centres: np.ndarray, twist: float = 0.0) -> np.ndarray:
    """Dense width sections; flat fitted ribbons follow the doll across their width."""
    c = dense_centres(centres)
    out = np.repeat(c[:, None, :], WIDTH_SAMPLES, axis=1)
    out[:, :, 1] += np.linspace(-STRAP_WIDTH / 2, STRAP_WIDTH / 2, WIDTH_SAMPLES)
    base = surface_w(c[:, 0], c[:, 1])[:, None]
    edge = surface_w(out[:, :, 0], out[:, :, 1])
    row = np.arange(len(c))[:, None]
    valid = (row >= CONTROL_ROWS[1]) & (row <= CONTROL_ROWS[5]) & np.isfinite(base) & np.isfinite(edge)
    out[:, :, 2] += np.where(valid, edge - base, 0.0)
    if twist:
        # The measured shoulder roll rotates the material's ordered width edges.
        weights = np.interp(np.arange(len(c)), CONTROL_ROWS, [0.0, 0.5, 1.0, 1.0, 0.5, 0.0, 0.0])
        theta = twist * weights[:, None]
        cc, ss = np.cos(theta), np.sin(theta)
        delta = out - c[:, None]
        out[:, :, 1] = c[:, None, 1] + cc * delta[:, :, 1] - ss * delta[:, :, 2]
        out[:, :, 2] = c[:, None, 2] + ss * delta[:, :, 1] + cc * delta[:, :, 2]
    return out


def polyline_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(np.asarray(points, float), axis=0), axis=1).sum())


def surface_samples(points: np.ndarray, spacing: float = 0.001) -> np.ndarray:
    """Sample the actual displayed triangles, including every edge, at <=1 mm spacing."""
    p = np.asarray(points, float)
    samples = []
    controls = CONTROL_ROWS if len(p) == LONGITUDINAL_SAMPLES else tuple(range(7))
    for row in range(controls[1], controls[5]):
        for col in range(p.shape[1] - 1):
            a, b, c, d = p[row, col], p[row, col + 1], p[row + 1, col + 1], p[row + 1, col]
            for triangle in (np.array([a, b, c]), np.array([a, c, d])):
                longest = max(np.linalg.norm(triangle[i] - triangle[j]) for i, j in ((0, 1), (1, 2), (2, 0)))
                n = max(1, int(math.ceil(longest / spacing)))
                bary = np.array([(i / n, j / n) for i in range(n + 1) for j in range(n + 1 - i)])
                samples.append(triangle[0] + bary[:, :1] * (triangle[1] - triangle[0]) + bary[:, 1:] * (triangle[2] - triangle[0]))
    return np.vstack(samples)


def surface_triangles(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, float)
    controls = CONTROL_ROWS if len(p) == LONGITUDINAL_SAMPLES else tuple(range(7))
    patch = p[controls[1]:controls[5] + 1]
    a, b = patch[:-1, :-1], patch[:-1, 1:]
    c, d = patch[1:, 1:], patch[1:, :-1]
    return np.stack((np.stack((a, b, c), axis=-2), np.stack((a, c, d), axis=-2)), axis=-3).reshape(-1, 3, 3)


def fold_height_for_length(centres: np.ndarray, material_length: float) -> float:
    """Geometric coupling for the material feed; its result is not scoring evidence."""
    c = np.array(centres, float, copy=True)
    if reference_length(c) >= material_length:
        return 0.0
    base = c[2, 2]
    low, high = 0.0, 0.10
    for _ in range(40):
        middle = (low + high) / 2
        c[2, 2] = base + middle
        if reference_length(c) < material_length:
            low = middle
        else:
            high = middle
    return (low + high) / 2


@dataclass(frozen=True)
class FitEvidence:
    routed: bool
    untwisted: bool
    against_body: bool
    material_conserved: bool
    fold_bound_pass: bool
    material_length_m: float
    fitted_length_m: float
    local_overlap_bound_m: float
    min_signed_gap_m: float
    max_signed_gap_m: float
    global_chord_overlap_bound_m: float
    reference_excess_m: float
    pinch: dict

    @property
    def passes(self) -> bool:
        return self.routed and self.untwisted and self.against_body and self.material_conserved and self.fold_bound_pass

    def record(self) -> dict:
        return dict(asdict(self), passes=self.passes)


def assess_ribbon(points: np.ndarray, side: int, tongue: np.ndarray, *, material_floor: float | None = None) -> FitEvidence:
    """Assess actual ribbon vertices; no command, mode, or 'tightened' flag is accepted.

    material_floor is only a fail-closed conservation check on the physical
    material feed model: flattening geometry cannot silently delete webbing.
    It never substitutes for measuring path length or contact from vertices.
    """
    p = np.asarray(points, float)
    if side not in (-1, 1) or p.ndim != 3 or p.shape[0] not in (7, LONGITUDINAL_SAMPLES) or p.shape[2] != 3 or p.shape[1] < 3 or p.shape[1] % 2 == 0 or not np.isfinite(p).all():
        return FitEvidence(False, False, False, False, False, math.inf, 0.0, math.inf, -math.inf, math.inf, math.inf, math.inf, {"status": "indeterminate"})
    c = p[:, p.shape[1] // 2]
    reference = fitted_centres(side, np.asarray(tongue, float))
    fitted = reference_length(reference)
    length = polyline_length(c)
    controls = CONTROL_ROWS if len(p) == LONGITUDINAL_SAMPLES else tuple(range(7))
    body_rows = range(controls[1], controls[5] + 1)
    patch = p[list(body_rows)]
    contact, gap_low, gap_high = certify_surface(surface_triangles(p), BODY, PENETRATION, SURFACE_GAP)
    shoulder = p[controls[1]:controls[3] + 1]
    routed = bool(np.all((shoulder[:, :, 0] >= 0.120 - 1e-6) & (shoulder[:, :, 0] <= 0.145))
                  and np.all(side * shoulder[:, :, 1] >= 0.015)
                  and np.all(side * shoulder[:, :, 1] <= 0.060)
                  and np.all(side * patch[:, :, 1] >= 0.009)
                  and side * c[-1, 1] > 0)
    width = patch[:, -1] - patch[:, 0]
    norms = np.linalg.norm(width, axis=1)
    untwisted = bool(np.all(norms >= STRAP_WIDTH * 0.8)
                     and np.all(width[:, 1] / np.maximum(norms, 1e-12) >= math.cos(math.radians(30))))
    # The prescribed fitted route is not a shortest-path lower bound.
    # Only the measured, clamped shoulder patch receives the chord proof.
    pinch = assess_patch(shoulder, BODY, PENETRATION)
    global_bound = max(0.0, length - float(np.linalg.norm(c[-1] - c[0]))) / 2.0
    conserved = bool(material_floor is None or (math.isfinite(material_floor) and abs(length - material_floor) <= MATERIAL_TOLERANCE))
    return FitEvidence(routed, untwisted, contact, conserved, pinch.passes,
                       length, fitted, pinch.overlap_upper_m, gap_low, gap_high,
                       global_bound, length - fitted, pinch.record())


def clip_at_rest(history, steps=20, limit=0.005) -> bool:
    return len(history) > steps and float(np.ptp(np.asarray(history[-(steps + 1):], float))) < limit


def assess_lap(points: np.ndarray, side: int, tongue: np.ndarray) -> bool:
    """Check the measured lap ribbon reaches its own hip and buckle anchor without crossing."""
    p = np.asarray(points, float)
    if p.shape != (2, 2, 3) or not np.isfinite(p).all() or side not in (-1, 1):
        return False
    ends = p.mean(axis=1)
    anchor = np.array([-0.015, side * 0.085, 0.030])
    return bool(np.linalg.norm(ends[0] - anchor) <= 0.003
                and np.linalg.norm(ends[1] - tongue) <= 0.003
                and np.all(side * p[:, :, 1] >= 0.003)
                and np.all((p[:, 1, 1] - p[:, 0, 1]) > 0))


def lap_length(side: int, tongue: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(tongue) - np.array([-0.015, side * 0.085, 0.030])))


def material_feed_limit(side: int, tongue: np.ndarray, budget: float) -> float:
    """Available feed before the shortest fitted upper-plus-lap route is consumed."""
    minimum = reference_length(fitted_centres(side, tongue)) + lap_length(side, tongue)
    return float(np.clip(budget - minimum, 0.0, SHOULDER_SLACK))
