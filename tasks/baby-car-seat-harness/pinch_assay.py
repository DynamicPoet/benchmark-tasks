"""A fixed-endpoint shoulder diagnostic, separate from a human pinch test.

The proof concerns inextensible longitudinal lanes of the measured triangle mesh.
The constructed U is an ideal geometric candidate, not woven-cloth mechanics or
thumb/forefinger retention. No material may enter through the clamped sections.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import numpy as np

from body_distance import certify_surface

OVERLAP = 0.002
BEND_RADIUS = 0.0005
ARC_STEPS = 16
PAD_WIDTH, PAD_LENGTH, PAD_THICKNESS = 0.006, 0.004, 0.002
PAD_CONTACT_TOLERANCE = 0.00025
NORMAL_OPPOSITION = 0.99


def lane_bounds(patch):
    """Bound every fixed material lane, including inside the displayed triangles.

    In a quad split a-b-c/a-c-d, a lane at width fraction a crosses the
    diagonal: its length is (1-a)*L_left+a*L_right. Its endpoint difference
    is also affine. max(L_left,L_right)-min|endpoint_difference| is therefore
    an upper bound on L-D over the entire width interval, without sampling it.
    """
    p = np.asarray(patch, float)
    lengths = np.linalg.norm(np.diff(p, axis=0), axis=2).sum(axis=0)
    delta = p[-1] - p[0]
    chords = np.linalg.norm(delta, axis=1)
    change = np.diff(delta, axis=0)
    fraction = np.clip(-np.sum(delta[:-1] * change, axis=1) /
                       np.maximum(np.sum(change * change, axis=1), 1e-30), 0, 1)
    minimum_chord = np.linalg.norm(delta[:-1] + fraction[:, None] * change, axis=1)
    bounds = np.maximum(0., (np.maximum(lengths[:-1], lengths[1:]) - minimum_chord) / 2) + 1e-9
    return lengths, chords, bounds


def triangles(points):
    p = np.asarray(points, float)
    a, b = p[:-1, :-1], p[:-1, 1:]
    c, d = p[1:, 1:], p[1:, :-1]
    return np.stack((np.stack((a, b, c), axis=-2), np.stack((a, c, d), axis=-2)), axis=-3).reshape(-1, 3, 3)


def pad_mesh(centre, along, width, outward):
    """Inner face touches the layer; the solid pad extends away from the gap."""
    vertices = np.array([centre + z * outward * PAD_THICKNESS +
                         x * along * PAD_LENGTH / 2 + y * width * PAD_WIDTH / 2
                         for z in (0., 1.) for x, y in ((-1., -1.), (-1., 1.), (1., 1.), (1., -1.))])
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
                      [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
                      [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]])
    return vertices, faces


@dataclass
class FoldWitness:
    valid: bool
    overlap_m: float
    min_leg_height_m: float
    layer_spacing_m: float
    material_error_m: float
    endpoint_error_m: float
    opposed_normal_dot: float
    pad_contact_error_m: float
    within_body_tolerance: bool
    min_body_gap_m: float
    pads_disjoint: bool
    pads_away_from_gap: bool
    pad_separation_lower_m: float
    points: np.ndarray = field(repr=False)
    pads: tuple = field(repr=False)

    def record(self):
        record = {k: v for k, v in asdict(self).items() if k not in ('points', 'pads')}
        record['strict_nonpenetration_certified'] = self.min_body_gap_m >= 0.
        return record


def make_witness(patch, bodies=(), penetration=0.002):
    """Construct parallel U legs with a finite connector, preserving each lane.

    For endpoints distance D and semicircle chord 2r, the complete candidate
    length is D+2h+C-2r. The sampled connector's measured C, not pi*r, is used.
    This preserves warp lengths, but does not certify transverse cloth metrics,
    self-contact dynamics, bending resistance, friction or finger retention.
    """
    p = np.asarray(patch, float)
    lengths = np.linalg.norm(np.diff(p, axis=0), axis=2).sum(axis=0)
    delta = p[-1] - p[0]
    chord = np.linalg.norm(delta, axis=1)
    if np.any(chord <= 2 * BEND_RADIUS):
        return None
    tangent = delta / chord[:, None]
    normal = np.array([0., 0., 1.]) - tangent * tangent[:, 2:3]
    norms = np.linalg.norm(normal, axis=1)
    if np.any(norms < 1e-9):
        return None
    normal /= norms[:, None]
    connector = 2 * ARC_STEPS * BEND_RADIUS * math.sin(math.pi / (2 * ARC_STEPS))
    height = (lengths - chord - connector + 2 * BEND_RADIUS) / 2
    if np.any(height < OVERLAP):
        return None
    middle = (p[0] + p[-1]) / 2
    a, b = middle - BEND_RADIUS * tangent, middle + BEND_RADIUS * tangent
    arc = np.array([middle + height[:, None] * normal + BEND_RADIUS *
                    (math.cos(theta) * tangent + math.sin(theta) * normal)
                    for theta in np.linspace(math.pi, 0., ARC_STEPS + 1)])
    candidate = np.concatenate((p[:1], a[None], arc, b[None], p[-1:]), axis=0)
    measured = np.linalg.norm(np.diff(candidate, axis=0), axis=2).sum(axis=0)
    material_error = float(np.max(abs(measured - lengths)))
    endpoint_error = float(max(np.linalg.norm(candidate[0] - p[0], axis=1).max(),
                               np.linalg.norm(candidate[-1] - p[-1], axis=1).max()))
    col = p.shape[1] // 2
    t, n = tangent[col], normal[col]
    # Orient the finite pad faces to the actual ruled layers, rather than
    # incorrectly treating a curved cross-width section as a flat +v strip.
    outgoing_width = (a[col + 1] + height[col + 1] / 2 * normal[col + 1]) - (a[col - 1] + height[col - 1] / 2 * normal[col - 1])
    incoming_width = (b[col + 1] + height[col + 1] / 2 * normal[col + 1]) - (b[col - 1] + height[col - 1] / 2 * normal[col - 1])
    na, nb = np.cross(normal[col], outgoing_width), np.cross(-normal[col], incoming_width)
    na /= max(np.linalg.norm(na), 1e-20)
    nb /= max(np.linalg.norm(nb), 1e-20)
    opposition = float(na @ nb)
    # Surface winding may reverse when a material section is reversed. Choose
    # the contact-face normals toward the other layer from actual geometry;
    # both finite pad solids then extend in the opposite direction.
    across = b[col] - a[col]
    fa = na if float(na @ across) > 0 else -na
    fb = nb if float(nb @ -across) > 0 else -nb
    away = bool(float((-fa) @ across) < 0 and float((-fb) @ -across) < 0)
    wa, wb = np.cross(fa, n), np.cross(-fb, n)
    wa /= max(np.linalg.norm(wa), 1e-20)
    wb /= max(np.linalg.norm(wb), 1e-20)
    pad_a = a[col] + height[col] / 2 * n
    pad_b = b[col] + height[col] / 2 * n
    pads = (pad_mesh(pad_a, n, wa, -fa), pad_mesh(pad_b, n, wb, -fb))
    # Exact separating-axis test for the two oriented rectangular pad solids.
    axes_a, axes_b = (n, wa, fa), (n, wb, fb)
    axes = list(axes_a) + list(axes_b) + [np.cross(x, y) for x in axes_a for y in axes_b]
    gaps = []
    for axis in axes:
        norm = np.linalg.norm(axis)
        if norm <= 1e-12:
            continue
        axis = axis / norm
        pa, pb = pads[0][0] @ axis, pads[1][0] @ axis
        gaps.append(max(float(pb.min() - pa.max()), float(pa.min() - pb.max())))
    pad_separation = max(gaps)
    disjoint = pad_separation > 1e-9
    # Three material lanes in the central pad region, each over two millimetres
    # of common height. Verify actual candidate surfaces, not an intended label.
    columns = (max(0, col - 1), col, min(p.shape[1] - 1, col + 1))
    worst_contact, overlap = 0., float('inf')
    contact_valid = True
    for base, centre, face_normal, width in ((a, pad_a, fa, wa), (b, pad_b, fb, wb)):
        for j in columns:
            start = float(n @ (base[j] - centre))
            end = float(n @ (base[j] + height[j] * normal[j] - centre))
            overlap = min(overlap, min(end, PAD_LENGTH / 2) - max(start, -PAD_LENGTH / 2))
            for level in (-OVERLAP / 2, OVERLAP / 2):
                along = (level - start) / float(n @ normal[j])
                sample = base[j] + along * normal[j]
                diff = sample - centre
                worst_contact = max(worst_contact, abs(float(face_normal @ diff)))
                contact_valid &= (0 <= along <= height[j] and abs(float(width @ diff)) <= PAD_WIDTH / 2 + 1e-9)
    body_clear, low = True, float('inf')
    if bodies:
        surfaces = np.concatenate([triangles(candidate)] + [v[f] for v, f in pads])
        body_clear, low, _ = certify_surface(surfaces, bodies, penetration, 1.0)
    valid = bool(material_error <= 1e-7 and endpoint_error <= 1e-9 and overlap >= OVERLAP
                 and contact_valid and worst_contact <= PAD_CONTACT_TOLERANCE
                 and opposition <= -NORMAL_OPPOSITION and body_clear and disjoint and away)
    return FoldWitness(valid, max(0., overlap), float(height.min()), float(np.linalg.norm(b[col] - a[col])),
                       material_error, endpoint_error, opposition, worst_contact,
                       bool(body_clear), low, disjoint, away, pad_separation, candidate, pads)


@dataclass
class PinchEvidence:
    status: str
    overlap_upper_m: float
    centre_overlap_upper_m: float
    patch_material_m: float
    patch_chord_m: float
    witness: FoldWitness | None = None

    @property
    def passes(self):
        return self.status == 'no_fold_certified'

    def record(self):
        return dict(status=self.status, scope='fixed_material_end_sections', external_feed_allowed=False,
                    human_finger_retention_validated=False, overlap_upper_m=self.overlap_upper_m,
                    centre_overlap_upper_m=self.centre_overlap_upper_m,
                    patch_material_m=self.patch_material_m, patch_chord_m=self.patch_chord_m,
                    witness=None if self.witness is None else self.witness.record())


def assess_patch(patch, bodies=(), penetration=0.002):
    p = np.asarray(patch, float)
    if (p.ndim != 3 or p.shape[0] < 2 or p.shape[1] < 3 or p.shape[1] % 2 != 1
            or p.shape[2] != 3 or not np.isfinite(p).all()):
        return PinchEvidence('indeterminate', math.inf, math.inf, math.inf, 0.)
    lengths, chords, bounds = lane_bounds(p)
    col = p.shape[1] // 2
    upper = float(bounds.max())
    centre_upper = max(0., float(lengths[col] - chords[col])) / 2 + 1e-9
    if upper < OVERLAP - 1e-9:
        return PinchEvidence('no_fold_certified', upper, centre_upper, float(lengths[col]), float(chords[col]))
    witness = make_witness(p, bodies, penetration)
    return PinchEvidence('fold_witness' if witness is not None and witness.valid else 'indeterminate',
                         upper, centre_upper, float(lengths[col]), float(chords[col]), witness)
