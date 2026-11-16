"""A full-size folding manual wheelchair as one USD articulation, plus the geometry the scorer measures.

Sizes follow a standard lightweight folding wheelchair with an 18 in (457 mm) seat: 24 in rear wheels with handrims,
8 in swivel casters, a seat 48 cm above the floor, flip-back desk armrests and an X cross-brace under the seat.
Pure Python and NumPy: it writes plain-text USD, so it runs before the simulator starts.

Chair coordinates (metres): origin on the floor below the midpoint of the rear-wheel hubs, +x towards the chair's
back, +y to the chair's right-hand side as seen from the front (the robot's left), +z up. The chair faces -x, so its
front, with the casters, is nearest the robot. Every body is authored at identity in these coordinates, in the
folded configuration, so a joint's zero is the folded chair and a body's local coordinates are chair coordinates.

Bodies: two side frames (``frame_l`` at +y, ``frame_r`` at -y), two cross-braces (``brace_l`` carries the +y seat
rail and pivots on the -y frame's lower rail, ``brace_r`` the reverse), two rear wheels, two caster forks and their
wheels, and four light fabric panels (seat and back halves). The articulation's tree runs
frame_l -> brace_r -> brace_l -> frame_r, so the X is a chain and needs no loop joint. The seat rails rest on
saddles on the side frames when the chair is open: that contact, not a joint limit, ends the opening.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

import usdkit as uk

HERE = Path(__file__).resolve().parent

# -- dimensions (m) ----------------------------------------------------------------------------------------------------

W_OPEN = 0.2286  # half the distance between the seat rails when open (18 in seat)
W_FOLD = 0.065  # the same, folded: 28 cm folded width over the handrims
Z_PIVOT = 0.14  # cross-brace pivots on the frames' lower rails
Z_HOUSING = 0.22  # front post's lower end, at the caster housing
X_LOWER = -0.27  # the lower rail starts behind the caster's swing
Z_RAIL_OPEN = 0.48  # seat-rail centre when open, resting in its saddles
R_RAIL = 0.0125  # seat rail tube radius (25 mm)
R_TUBE = 0.011  # frame tube radius (22 mm)
SADDLE_TOP = Z_RAIL_OPEN - R_RAIL
L_BRACE = math.hypot(Z_RAIL_OPEN - Z_PIVOT, 2 * W_OPEN)  # pivot to seat-rail centre
Z_RAIL_FOLD = Z_PIVOT + math.sqrt(L_BRACE**2 - (2 * W_FOLD) ** 2)
Z_PIN_FOLD = (Z_PIVOT + Z_RAIL_FOLD) / 2
X_MID = -0.15  # the X's plane, a little ahead of the rear axle
BRACE_DX = 0.013  # each brace sits this far in front of or behind the X's plane, so the two pass each other
X_RAIL = (-0.44, -0.02)  # seat rail ends
X_FRONT = -0.43  # front posts and caster swivels
X_CANE = 0.04  # back canes
Z_CANE_TOP = 0.89
HANDLE_END = (0.15, 0.93)  # push-handle grip end (x, z)
R_WHEEL = 0.305  # 24 in rear wheel
AXLE_OUT = 0.04  # rear hub outboard of the frame plane
TIRE_W = 0.030
R_HANDRIM = 0.270
HANDRIM_OUT = 0.025  # handrim outboard of the wheel's plane
R_CASTER = 0.1016  # 8 in caster
CASTER_W = 0.030
CASTER_TRAIL = 0.045  # caster axle behind its swivel
Z_SWIVEL = 0.205  # fork crown
SADDLES_X = (-0.41, -0.06)
SADDLE_LEN = 0.045
SADDLE_FLANK, SADDLE_T = 0.030, 0.004  # each 45-degree flank of the V cradle: width, thickness
SADDLE_APEX = SADDLE_TOP - (R_RAIL * math.sqrt(2) - R_RAIL)  # the V's bottom line; a seated rail's bottom is SADDLE_TOP
ARM_PAD = dict(x=(-0.22, 0.03), y=(0.02, 0.07), z=(0.72, 0.76))  # flip-back desk armrest pad, y outboard of the frame
SEAT_HALF = W_OPEN  # each seat half spans the open half-width
BACK_Z = (0.53, 0.88)  # back upholstery
SEAT_DEPTH = (-0.43, -0.03)

MASS = {  # kg; about 13 kg in all
    "frame": 2.85,  # tubes, axle plate, caster housing, saddles, armrest, clothing guard, brake, hardware
    "brace": 0.85,  # cross-brace tube and seat rail
    "wheel": 2.0,  # 24 in spoked wheel, tyre and handrim
    "fork": 0.25,
    "caster": 0.35,
    "seat_half": 0.15,
    "back_half": 0.15,
}
TOTAL_MASS = 2 * sum(MASS.values())

SIDES = {"l": 1.0, "r": -1.0}

# -- small vector helpers ----------------------------------------------------------------------------------------------


def _v(*a) -> np.ndarray:
    return np.array(a, dtype=float)


def quat_from_z(d) -> tuple[float, float, float, float]:
    """(w, x, y, z) turning +z onto direction d."""
    d = np.asarray(d, float)
    d = d / np.linalg.norm(d)
    axis = np.cross([0.0, 0.0, 1.0], d)
    s, c = np.linalg.norm(axis), d[2]
    if s < 1e-12:
        return (1.0, 0.0, 0.0, 0.0) if c > 0 else (0.0, 1.0, 0.0, 0.0)
    axis /= s
    ang = math.atan2(s, c)
    return (math.cos(ang / 2), *(axis * math.sin(ang / 2)))


def quat_to_matrix_wxyz(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def matrix_to_quat_wxyz(r: np.ndarray) -> tuple[float, float, float, float]:
    tr = np.trace(r)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = (0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = ((r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s)
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = ((r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s)
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = ((r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s)
    q = np.array(q)
    return tuple(q / np.linalg.norm(q))


def rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


# -- mass properties ---------------------------------------------------------------------------------------------------


@dataclass
class MassModel:
    """Parts (mass, centre, inertia about own centre) summed into one body's mass, centre of mass and principal
    inertia."""

    parts: list = field(default_factory=list)

    def rod(self, m, a, b, r=0.011):
        a, b = _v(*a), _v(*b)
        d = b - a
        length = np.linalg.norm(d)
        u = d / max(length, 1e-9)
        i_ax, i_tr = 0.5 * m * r * r, m * (3 * r * r + length * length) / 12
        inertia = i_tr * np.eye(3) + (i_ax - i_tr) * np.outer(u, u)
        self.parts.append((m, (a + b) / 2, inertia))

    def box(self, m, center, size):
        sx, sy, sz = size
        inertia = np.diag([sy * sy + sz * sz, sx * sx + sz * sz, sx * sx + sy * sy]) * m / 12
        self.parts.append((m, _v(*center), inertia))

    def ring(self, m, center, r, axis):
        u = _v(*axis) / np.linalg.norm(axis)
        i_ax, i_tr = m * r * r, 0.5 * m * r * r
        self.parts.append((m, _v(*center), i_tr * np.eye(3) + (i_ax - i_tr) * np.outer(u, u)))

    def disk(self, m, center, r, axis):
        u = _v(*axis) / np.linalg.norm(axis)
        i_ax, i_tr = 0.5 * m * r * r, 0.25 * m * r * r
        self.parts.append((m, _v(*center), i_tr * np.eye(3) + (i_ax - i_tr) * np.outer(u, u)))

    def scaled_to(self, total: float) -> "MassModel":
        k = total / sum(p[0] for p in self.parts)
        return MassModel([(m * k, c, inertia * k) for m, c, inertia in self.parts])

    def properties(self):
        m = sum(p[0] for p in self.parts)
        com = sum(p[0] * p[1] for p in self.parts) / m
        inertia = np.zeros((3, 3))
        for mi, c, ii in self.parts:
            d = c - com
            inertia += ii + mi * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        vals, vecs = np.linalg.eigh(inertia)
        if np.linalg.det(vecs) < 0:
            vecs[:, 0] *= -1
        return m, com, vals, matrix_to_quat_wxyz(vecs)


# -- the parts ---------------------------------------------------------------------------------------------------------


@dataclass
class Collider:
    kind: str  # capsule, box, hull
    center: np.ndarray
    quat: tuple  # wxyz, turning the shape's +z (capsule axis) or box axes
    size: tuple  # capsule: (radius, height); box: (sx, sy, sz); hull: points [n, 3] (absolute)
    material: str = "metal"


@dataclass
class Body:
    name: str
    mass: MassModel = field(default_factory=MassModel)
    colliders: list = field(default_factory=list)
    visuals: list = field(default_factory=list)  # (name, Mesh, look)
    probes: list = field(default_factory=list)  # solids for the scorer's distance checks only (no collision)

    def capsule(self, a, b, r, material="metal"):
        a, b = _v(*a), _v(*b)
        d = b - a
        self.colliders.append(Collider("capsule", (a + b) / 2, quat_from_z(d), (r, float(np.linalg.norm(d))), material))

    def box(self, center, size, material="metal", quat=(1.0, 0.0, 0.0, 0.0)):
        self.colliders.append(Collider("box", _v(*center), quat, tuple(size), material))

    def hull(self, points, material="rubber"):
        self.colliders.append(Collider("hull", np.zeros(3), (1.0, 0.0, 0.0, 0.0), np.asarray(points, float), material))


def _wheel_points(center, radius, width, segments=48) -> np.ndarray:
    """Points of a cylinder (axis y) for a convex-hull collider."""
    # half a segment's turn puts a flat face, not a corner, at the bottom, so the wheel stands still
    th = np.linspace(0, 2 * np.pi, segments, endpoint=False) + np.pi / segments
    ring = np.c_[radius * np.cos(th), np.zeros(segments), radius * np.sin(th)]
    return np.vstack([ring + center + (0, -width / 2, 0), ring + center + (0, width / 2, 0)])


def _y_cylinder(radius, width, center, segments=32, bevel=0.0) -> uk.Mesh:
    return uk.cylinder(radius, width, center=center, axis="y", segments=segments, bevel=bevel)


def _torus(major, minor, center, segments=48, sides=12) -> uk.Mesh:
    """A ring about the y axis."""
    prof = [(major + minor * math.cos(a), minor * math.sin(a)) for a in np.linspace(-np.pi, np.pi, sides + 1)]
    prof = prof[:-1]
    th = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    verts, faces = [], []
    for r, z in prof:
        verts += [(r * math.cos(t), r * math.sin(t), z) for t in th]
    n = len(prof)
    for k in range(n):
        k2 = (k + 1) % n
        for i in range(segments):
            j = (i + 1) % segments
            a, b, c, d = k * segments + i, k * segments + j, k2 * segments + j, k2 * segments + i
            faces += [(a, c, b), (a, d, c)]
    m = uk.Mesh(np.array(verts), np.array(faces))
    return m.turned(-90.0, "x").moved(center)


def _tube(points, r, sides=12) -> uk.Mesh:
    p = np.asarray(points, float)
    # densify straight runs so the tube keeps its radius through bends
    dense = [p[0]]
    for a, b in zip(p[:-1], p[1:]):
        n = max(1, int(np.linalg.norm(b - a) / 0.03))
        dense += [a + (b - a) * (k / n) for k in range(1, n + 1)]
    return uk.tube(np.array(dense), r, sides)


def _bend(a, corner, b, r=0.04, n=6):
    """Polyline a -> corner -> b with the corner rounded (radius about r)."""
    a, corner, b = _v(*a), _v(*corner), _v(*b)
    u1 = (corner - a) / np.linalg.norm(corner - a)
    u2 = (b - corner) / np.linalg.norm(b - corner)
    p1, p2 = corner - u1 * r, corner + u2 * r
    pts = [a]
    for k in range(n + 1):
        t = k / n
        pts.append((1 - t) ** 2 * p1 + 2 * (1 - t) * t * corner + t * t * p2)
    pts.append(b)
    return pts


LOOKS = {
    "frame": uk.Look(color=(0.05, 0.12, 0.36), metallic=0.55, roughness=0.32),  # blue powder-coated aluminium
    "chrome": uk.Look(color=(0.80, 0.80, 0.82), metallic=1.0, roughness=0.18),
    "steel": uk.Look(color=(0.55, 0.56, 0.58), metallic=0.9, roughness=0.35),
    "tyre": uk.Look(color=(0.025, 0.025, 0.028), roughness=0.85),
    "black_plastic": uk.Look(color=(0.03, 0.03, 0.035), roughness=0.55),
    "grey_plastic": uk.Look(color=(0.18, 0.18, 0.19), roughness=0.5),
    "vinyl": uk.Look(color=(0.018, 0.018, 0.02), roughness=0.45),
    "upholstery": uk.Look(color=(0.02, 0.02, 0.025), roughness=0.75),
    "foam_grip": uk.Look(color=(0.03, 0.03, 0.03), roughness=0.9),
}


def side_frame(side: str) -> Body:
    s = SIDES[side]
    y = s * W_FOLD
    out = s  # outward sign
    b = Body(f"frame_{side}")
    m = b.mass
    # tubes: lower rail, front post, back cane with push handle
    lower = [(X_LOWER, y, Z_PIVOT), (X_CANE, y, Z_PIVOT)]
    drop = [(X_FRONT, y, Z_HOUSING), (X_LOWER, y, Z_PIVOT)]  # front post down to the lower rail
    front = [(X_FRONT, y, Z_HOUSING - 0.02), (X_FRONT, y, SADDLE_TOP - 0.012)]
    cane = _bend((X_CANE, y, Z_PIVOT), (X_CANE, y, HANDLE_END[1]), (HANDLE_END[0], y, HANDLE_END[1]), r=0.05)
    b.capsule(*lower, R_TUBE)
    b.capsule(*drop, R_TUBE)
    b.capsule(*front, R_TUBE)
    b.capsule((X_CANE, y, Z_PIVOT), (X_CANE, y, HANDLE_END[1] - 0.04), R_TUBE)
    b.capsule((X_CANE + 0.03, y, HANDLE_END[1]), (HANDLE_END[0], y, HANDLE_END[1]), 0.014)
    m.rod(0.20, *lower)
    m.rod(0.10, *drop)
    m.rod(0.18, *front)
    m.rod(0.42, (X_CANE, y, Z_PIVOT), (X_CANE, y, HANDLE_END[1]))
    m.rod(0.10, (X_CANE, y, HANDLE_END[1]), (HANDLE_END[0], y, HANDLE_END[1]))
    b.visuals += [("lower_rail", _tube(lower, R_TUBE), "frame"), ("front_post", _tube(front, R_TUBE), "frame"),
                  ("lower_drop", _tube(drop, R_TUBE), "frame"),
                  ("cane", _tube(cane, R_TUBE), "frame"),
                  ("grip", uk.cylinder(0.0145, 0.11, center=(HANDLE_END[0] - 0.05, y, HANDLE_END[1]), axis="x",
                                       segments=16, bevel=0.004), "foam_grip")]
    # caster housing (the fork's stem turns inside it)
    b.capsule((X_FRONT, y, Z_SWIVEL + 0.012), (X_FRONT, y, Z_HOUSING + 0.045), 0.019)
    m.rod(0.20, (X_FRONT, y, Z_SWIVEL), (X_FRONT, y, Z_HOUSING + 0.05), 0.02)
    b.visuals.append(("caster_housing", uk.cylinder(0.019, 0.07, center=(X_FRONT, y, Z_SWIVEL + 0.035), segments=20,
                                                    bevel=0.003), "steel"))
    # axle plate: carries the rear wheel's axle, outboard of the frame
    plate_c = (0.012, y + out * 0.016, 0.285)
    b.box(plate_c, (0.10, 0.008, 0.16))
    m.box(0.35, plate_c, (0.10, 0.008, 0.16))
    b.visuals.append(("axle_plate", uk.box((0.10, 0.008, 0.16), center=plate_c, round_xy=0.0, bevel=0.003), "steel"))
    axle = [(0.0, y, R_WHEEL), (0.0, y + out * (AXLE_OUT - 0.005), R_WHEEL)]
    b.visuals.append(("axle", _tube(axle, 0.008, 10), "chrome"))
    m.rod(0.10, *axle, 0.008)
    # saddles: V-shaped plastic cradles the seat rail drops into (two 45-degree flanks); a seated rail touches both
    for k, sx in enumerate(SADDLES_X):
        apex = _v(sx, y, SADDLE_APEX)
        flank = []
        for side_k in (1, -1):
            up = _v(0, side_k * math.cos(math.pi / 4), math.sin(math.pi / 4))  # along the flank, up and out
            inner = _v(0, -side_k * math.sin(math.pi / 4), math.cos(math.pi / 4))  # its face's normal, inward-up
            c = apex + up * SADDLE_FLANK / 2 - inner * SADDLE_T / 2
            q = (math.cos(side_k * math.pi / 8), math.sin(side_k * math.pi / 8), 0.0, 0.0)
            b.box(c, (SADDLE_LEN, SADDLE_FLANK, SADDLE_T), material="metal", quat=q)
            mesh = uk.box((SADDLE_LEN, SADDLE_FLANK, SADDLE_T), bevel=0.001)
            mesh = uk.Mesh(mesh.v @ rot_x(side_k * math.pi / 4).T + c, mesh.f)
            flank.append(mesh)
        block = uk.box((SADDLE_LEN, 0.03, 0.016), center=(sx, y, SADDLE_APEX - 0.012), bevel=0.002)
        b.box((sx, y, SADDLE_APEX - 0.012), (SADDLE_LEN, 0.026, 0.014))
        m.box(0.06, (sx, y, SADDLE_APEX - 0.01), (SADDLE_LEN, 0.03, 0.02))
        b.visuals.append((f"saddle{k}", uk.join(block, *flank), "black_plastic"))
    rear_bracket = [(SADDLES_X[1], y, SADDLE_TOP - 0.02), (X_CANE, y, SADDLE_TOP - 0.02)]
    b.capsule(*rear_bracket, 0.009)
    b.visuals.append(("saddle_bracket", _tube(rear_bracket, 0.009), "frame"))
    m.rod(0.05, *rear_bracket)
    # flip-back desk armrest, cantilevered from the back cane
    ax, ay, az = ARM_PAD["x"], ARM_PAD["y"], ARM_PAD["z"]
    pad_c = ((ax[0] + ax[1]) / 2, y + out * (ay[0] + ay[1]) / 2, (az[0] + az[1]) / 2)
    pad_s = (ax[1] - ax[0], ay[1] - ay[0], az[1] - az[0])
    b.box(pad_c, pad_s, material="vinyl")
    m.box(0.30, pad_c, pad_s)
    b.visuals.append(("arm_pad", uk.box(pad_s, center=pad_c, bevel=0.008, round_xy=0.012), "vinyl"))
    arm_bar = [(ax[0] + 0.02, y + out * 0.045, az[0] - 0.012), (X_CANE - 0.005, y + out * 0.045, az[0] - 0.012)]
    arm_post = [(X_CANE - 0.005, y + out * 0.045, az[0] - 0.012), (X_CANE - 0.005, y + out * 0.045, az[0] - 0.12),
                (X_CANE - 0.005, y + out * 0.012, az[0] - 0.14)]
    b.capsule(arm_bar[0], arm_bar[1], 0.010)
    b.visuals += [("arm_bar", _tube(arm_bar, 0.010), "frame"), ("arm_post", _tube(arm_post, 0.011), "frame")]
    m.rod(0.15, *arm_bar)
    m.rod(0.10, arm_post[0], arm_post[2])
    # clothing guard: a black plastic panel between the wheel and the seat, on the armrest
    guard_c = (-0.07, y + out * 0.024, 0.60)
    b.box(guard_c, (0.24, 0.004, 0.20), material="vinyl")
    m.box(0.20, guard_c, (0.24, 0.004, 0.20))
    b.visuals.append(("clothing_guard", uk.box((0.24, 0.004, 0.20), center=guard_c, round_xy=0.0, bevel=0.0015),
                      "black_plastic"))
    # parking brake (lever and shoe ahead of the wheel); fixed, released
    brake_c = (-0.30, y + out * 0.03, 0.33)
    m.box(0.25, brake_c, (0.08, 0.03, 0.04))
    lever = [(-0.29, y + out * 0.03, 0.33), (-0.25, y + out * 0.03, 0.44)]
    b.visuals += [("brake_body", uk.box((0.07, 0.025, 0.035), center=brake_c, bevel=0.004), "black_plastic"),
                  ("brake_lever", _tube(lever, 0.007, 8), "steel"),
                  ("brake_knob", uk.cylinder(0.012, 0.03, center=lever[1], segments=12), "black_plastic")]
    brake_arm = [(X_FRONT, y, 0.33), (-0.30, y + out * 0.03, 0.33)]
    b.visuals.append(("brake_bar", _tube(brake_arm, 0.009), "frame"))
    m.rod(0.15, *brake_arm)
    b.mass = m.scaled_to(MASS["frame"])
    return b


def brace(side: str) -> Body:
    """brace_l carries the +y seat rail and pivots on frame_r's lower rail (and the reverse)."""
    s = SIDES[side]
    x = X_MID - s * BRACE_DX
    b = Body(f"brace_{side}")
    m = b.mass
    rail = [(X_RAIL[0], s * W_FOLD, Z_RAIL_FOLD), (X_RAIL[1], s * W_FOLD, Z_RAIL_FOLD)]
    b.capsule(*rail, R_RAIL, material="rail")
    m.rod(0.30, *rail, R_RAIL)
    diag = [(x, -s * W_FOLD, Z_PIVOT), (x, s * W_FOLD, Z_RAIL_FOLD)]
    d = _v(*diag[1]) - _v(*diag[0])
    u = d / np.linalg.norm(d)
    # the diagonal stops short of both ends: the pivot sleeve and the seat rail carry it
    b.capsule(_v(*diag[0]) + u * 0.04, _v(*diag[1]) - u * 0.03, R_TUBE)
    m.rod(0.30, *diag)
    sleeve = [(x - 0.035, -s * W_FOLD, Z_PIVOT), (x + 0.035, -s * W_FOLD, Z_PIVOT)]
    m.rod(0.10, *sleeve, 0.016)
    # a short gusset joins the diagonal to the seat rail
    gus = [_v(*diag[1]) - u * 0.09, (x + 0.06, s * W_FOLD, Z_RAIL_FOLD - 0.004)]
    m.rod(0.05, *gus)
    end_caps = [uk.cylinder(R_RAIL + 0.001, 0.02, center=(xe, s * W_FOLD, Z_RAIL_FOLD), axis="x", segments=16,
                            bevel=0.004) for xe in (X_RAIL[0] + 0.005, X_RAIL[1] - 0.005)]
    b.visuals += [("seat_rail", _tube(rail, R_RAIL, 14), "frame"), ("diagonal", _tube(diag, R_TUBE), "frame"),
                  ("sleeve", _tube(sleeve, 0.016, 14), "steel"), ("gusset", _tube(gus, 0.008, 8), "frame"),
                  ("rail_caps", uk.join(*end_caps), "black_plastic")]
    b.mass = m.scaled_to(MASS["brace"])
    return b


def rear_wheel(side: str) -> Body:
    s = SIDES[side]
    hub = _v(0.0, s * (W_FOLD + AXLE_OUT), R_WHEEL)
    b = Body(f"wheel_{side}")
    b.hull(_wheel_points(hub, R_WHEEL, TIRE_W, 64), material="rubber")
    m = b.mass
    m.ring(1.10, hub, R_WHEEL - 0.02, (0, 1, 0))  # rim and tyre
    hr = hub + (0, s * HANDRIM_OUT, 0)
    m.ring(0.40, hr, R_HANDRIM, (0, 1, 0))
    m.disk(0.50, hub, 0.10, (0, 1, 0))  # hub and spokes
    tyre = _torus(R_WHEEL - 0.0155, 0.0155, hub, 64, 12)
    rim = uk.join(_torus(R_WHEEL - 0.035, 0.008, hub + (0, 0.008, 0), 64, 8),
                  _torus(R_WHEEL - 0.035, 0.008, hub + (0, -0.008, 0), 64, 8))
    handrim = _torus(R_HANDRIM, 0.0095, hr, 64, 10)
    spokes = []
    for k in range(24):
        a = 2 * np.pi * k / 24
        side_off = 0.022 if k % 2 else -0.022
        p0 = hub + (0.03 * math.cos(a), side_off, 0.03 * math.sin(a))
        p1 = hub + ((R_WHEEL - 0.04) * math.cos(a + 0.25), 0.0, (R_WHEEL - 0.04) * math.sin(a + 0.25))
        spokes.append(uk.tube(np.array([p0, p1]), 0.0011, 4))
    tabs = []
    for k in range(6):
        a = 2 * np.pi * k / 6
        p0 = hub + ((R_WHEEL - 0.04) * math.cos(a), s * 0.002, (R_WHEEL - 0.04) * math.sin(a))
        p1 = hub + (R_HANDRIM * math.cos(a), s * HANDRIM_OUT, R_HANDRIM * math.sin(a))
        tabs.append(uk.tube(np.array([p0, p1]), 0.004, 6))
    b.visuals += [("tyre", tyre, "tyre"), ("rim", rim, "chrome"), ("handrim", handrim, "chrome"),
                  ("spokes", uk.join(*spokes), "chrome"), ("handrim_tabs", uk.join(*tabs), "chrome"),
                  ("hub", _y_cylinder(0.032, 0.06, hub, 20, 0.006), "steel")]
    b.mass = m.scaled_to(MASS["wheel"])
    return b


def caster_fork(side: str) -> Body:
    s = SIDES[side]
    y = s * W_FOLD
    b = Body(f"fork_{side}")
    m = b.mass
    ax = X_FRONT + CASTER_TRAIL
    crown_c = (X_FRONT + CASTER_TRAIL / 2, y, Z_SWIVEL - 0.008)
    m.box(0.10, crown_c, (0.08, 0.05, 0.012))
    blades = []
    for side_off in (-1, 1):
        a = (X_FRONT + 0.01, y + side_off * 0.024, Z_SWIVEL - 0.01)
        c = (ax, y + side_off * 0.024, R_CASTER)
        m.rod(0.06, a, c, 0.006)
        blades.append(_tube([a, c], 0.007, 8))
    stem = [(X_FRONT, y, Z_SWIVEL - 0.01), (X_FRONT, y, Z_SWIVEL + 0.06)]
    m.rod(0.03, *stem, 0.008)
    b.visuals += [("crown", uk.box((0.075, 0.056, 0.012), center=crown_c, bevel=0.003, round_xy=0.01), "steel"),
                  ("blades", uk.join(*blades), "steel"), ("stem", _tube(stem, 0.008, 10), "chrome")]
    b.box(crown_c, (0.06, 0.05, 0.012))
    b.mass = m.scaled_to(MASS["fork"])
    return b


def caster_wheel(side: str) -> Body:
    s = SIDES[side]
    c = _v(X_FRONT + CASTER_TRAIL, s * W_FOLD, R_CASTER)
    b = Body(f"caster_{side}")
    b.hull(_wheel_points(c, R_CASTER, CASTER_W, 40), material="rubber")
    m = b.mass
    m.disk(0.35, c, R_CASTER, (0, 1, 0))
    tyre = _torus(R_CASTER - 0.014, 0.014, c, 40, 10)
    hub = _y_cylinder(R_CASTER - 0.026, 0.026, c, 28, 0.004)
    b.visuals += [("tyre", tyre, "grey_plastic"), ("hub", hub, "black_plastic")]
    b.mass = m.scaled_to(MASS["caster"])
    return b


# -- fabric panels (seat and back halves, hinged on the seat rails and the canes) -----------------------------------------


def seat_fold_angle(rail_gap: float) -> float:
    """How far each seat half hangs below level (rad) for a gap between the seat rails' centres: the two halves
    meet in the middle, as the slack sling does."""
    return math.acos(min(1.0, max(-1.0, (rail_gap / 2) / SEAT_HALF)))


def back_fold_angle(cane_gap: float) -> float:
    return math.acos(min(1.0, max(-1.0, (cane_gap / 2) / SEAT_HALF)))


FOLD_SEAT = seat_fold_angle(2 * W_FOLD)
FOLD_BACK = back_fold_angle(2 * W_FOLD)


def seat_half(side: str) -> Body:
    """Seat half hinged on its seat rail, hanging at FOLD_SEAT below level when folded (authored that way)."""
    s = SIDES[side]
    b = Body(f"seat_{side}")
    hinge = _v(0.0, s * W_FOLD, Z_RAIL_FOLD + R_RAIL)
    panel = uk.box((SEAT_DEPTH[1] - SEAT_DEPTH[0], SEAT_HALF, 0.006),
                   center=((SEAT_DEPTH[0] + SEAT_DEPTH[1]) / 2, -s * SEAT_HALF / 2, 0.0), bevel=0.002)
    wrap = uk.cylinder(R_RAIL + 0.004, SEAT_DEPTH[1] - SEAT_DEPTH[0], center=((SEAT_DEPTH[0] + SEAT_DEPTH[1]) / 2, 0,
                                                                                -R_RAIL), axis="x", segments=14)
    r = rot_x(s * FOLD_SEAT)
    mesh = uk.join(panel, wrap)
    mesh = uk.Mesh(mesh.v @ r.T + hinge, mesh.f)
    b.visuals.append(("sling", mesh, "upholstery"))
    b.probes.append(Collider("box", hinge + r @ _v((SEAT_DEPTH[0] + SEAT_DEPTH[1]) / 2, -s * SEAT_HALF / 2, 0.0),
                             matrix_to_quat_wxyz(r), (SEAT_DEPTH[1] - SEAT_DEPTH[0], SEAT_HALF, 0.006)))
    b.mass.box(MASS["seat_half"], hinge + r @ _v((SEAT_DEPTH[0] + SEAT_DEPTH[1]) / 2, -s * SEAT_HALF / 2, 0),
               (0.4, 0.2, 0.01))
    return b


def back_half(side: str) -> Body:
    s = SIDES[side]
    b = Body(f"back_{side}")
    hinge = _v(X_CANE, s * W_FOLD, 0.0)
    h = BACK_Z[1] - BACK_Z[0]
    panel = uk.box((0.006, SEAT_HALF, h), center=(0.012, -s * SEAT_HALF / 2, (BACK_Z[0] + BACK_Z[1]) / 2), bevel=0.002)
    wrap = uk.cylinder(R_TUBE + 0.004, h, center=(0.0, 0.0, (BACK_Z[0] + BACK_Z[1]) / 2), segments=14)
    r = rot_z(s * FOLD_BACK)
    mesh = uk.join(panel, wrap)
    mesh = uk.Mesh(mesh.v @ r.T + hinge, mesh.f)
    b.visuals.append(("back", mesh, "upholstery"))
    b.probes.append(Collider("box", hinge + r @ _v(0.012, -s * SEAT_HALF / 2, (BACK_Z[0] + BACK_Z[1]) / 2),
                             matrix_to_quat_wxyz(r), (0.006, SEAT_HALF, h)))
    b.mass.box(MASS["back_half"], hinge + r @ _v(0.012, -s * SEAT_HALF / 2, (BACK_Z[0] + BACK_Z[1]) / 2),
               (0.01, 0.2, 0.35))
    return b


# -- joints ------------------------------------------------------------------------------------------------------------


@dataclass
class Joint:
    name: str
    body0: str
    body1: str
    pos: tuple
    axis: str
    limits: tuple | None = None  # degrees
    drive: bool = False  # an angular drive: the fabric panels follow the frames, the X's bushings damp


def joints() -> list[Joint]:
    j = [
        # the X: frame_l -> brace_r -> brace_l -> frame_r; folded is each joint's zero and its stop
        Joint("pivot_l", "frame_l", "brace_r", (X_MID + BRACE_DX, W_FOLD, Z_PIVOT), "X", (0.0, 50.0), drive=True),
        Joint("centre_pin", "brace_r", "brace_l", (X_MID, 0.0, Z_PIN_FOLD), "X", drive=True),
        Joint("pivot_r", "brace_l", "frame_r", (X_MID - BRACE_DX, -W_FOLD, Z_PIVOT), "X", (0.0, 50.0), drive=True),
    ]
    for side, s in SIDES.items():
        y = s * W_FOLD
        j += [
            Joint(f"axle_{side}", f"frame_{side}", f"wheel_{side}", (0.0, s * (W_FOLD + AXLE_OUT), R_WHEEL), "Y"),
            Joint(f"swivel_{side}", f"frame_{side}", f"fork_{side}", (X_FRONT, y, Z_SWIVEL), "Z"),
            Joint(f"caster_axle_{side}", f"fork_{side}", f"caster_{side}", (X_FRONT + CASTER_TRAIL, y, R_CASTER), "Y"),
            Joint(f"seat_hinge_{side}", f"brace_{side}", f"seat_{side}", (0.0, y, Z_RAIL_FOLD + R_RAIL), "X", drive=True),
            Joint(f"back_hinge_{side}", f"frame_{side}", f"back_{side}", (X_CANE, y, 0.0), "Z", drive=True),
        ]
    return j


# The back upholstery only pulls: it ties the back canes together at its top and bottom edges, so the side frames
# can lean apart only until it is taut. Fully open it is just taut. Modelled as two distance limits between the frames
# (D6 loop joints outside the articulation's tree, each with only a distance limit).
BACK_STRAP_Z = (BACK_Z[0] + 0.02, BACK_Z[1] - 0.02)
BACK_STRAP_LEN = 2 * W_OPEN + 0.004


def straps() -> list[tuple[str, tuple, tuple, float]]:
    return [(f"back_strap_{k}", (X_CANE, W_FOLD, z), (X_CANE, -W_FOLD, z), BACK_STRAP_LEN)
            for k, z in enumerate(BACK_STRAP_Z)]


def _strap_usda(name, p0, p1, length) -> str:
    return f"""
    def PhysicsJoint "{name}" (
        prepend apiSchemas = ["PhysicsLimitAPI:distance"]
    )
    {{
        rel physics:body0 = </{ROOT}/frame_l>
        rel physics:body1 = </{ROOT}/frame_r>
        point3f physics:localPos0 = {_f3(p0)}
        quatf physics:localRot0 = (1, 0, 0, 0)
        point3f physics:localPos1 = {_f3(p1)}
        quatf physics:localRot1 = (1, 0, 0, 0)
        float limit:distance:physics:low = 0
        float limit:distance:physics:high = {length:.6f}
        bool physics:excludeFromArticulation = 1
    }}"""


FABRIC_JOINTS = tuple(f"{k}_hinge_{side}" for k in ("seat", "back") for side in SIDES)


@lru_cache(maxsize=1)
def bodies() -> tuple[Body, ...]:
    out = [side_frame("l"), brace("r"), brace("l"), side_frame("r")]
    for side in SIDES:
        out += [rear_wheel(side), caster_fork(side), caster_wheel(side), seat_half(side), back_half(side)]
    return tuple(out)


# -- USD ---------------------------------------------------------------------------------------------------------------

ROOT = "Wheelchair"
PHYSICS_MATERIALS = {  # static, dynamic friction
    "rubber": (0.55, 0.55),  # rubber tyres on vinyl flooring (one coefficient: no stick-slip jump)
    "metal": (0.4, 0.3),
    "rail": (0.8, 0.7),  # aluminium seat rail against the gripper's rubber pads (averaged with the pads' 0.5)
    "vinyl": (0.6, 0.5),
}
FILTERED = {  # bodies that never touch in use; filtering them keeps tight clearances from fighting the solver
    "caster_l": ["frame_l"], "caster_r": ["frame_r"], "wheel_l": ["brace_l", "brace_r"],
    "wheel_r": ["brace_l", "brace_r"], "fork_l": ["brace_l", "brace_r"], "fork_r": ["brace_l", "brace_r"],
}


def _f3(v) -> str:
    return "(" + ", ".join(f"{float(x):.6f}" for x in v) + ")"


def _collider_usda(name: str, c: Collider) -> str:
    api = '"PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"'
    common = f"""
            rel material:binding:physics = </{ROOT}/PhysicsMaterials/{c.material}>
            uniform token purpose = "guide"
            float physxCollision:contactOffset = 0.004
            float physxCollision:restOffset = 0"""
    if c.kind == "capsule":
        r, h = c.size
        return f"""
        def Capsule "{name}" (
            prepend apiSchemas = [{api}]
        )
        {{{common}
            uniform token axis = "Z"
            double radius = {r:.6f}
            double height = {h:.6f}
            double3 xformOp:translate = {_f3(c.center)}
            quatf xformOp:orient = {_f3(c.quat)}
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
        }}"""
    if c.kind == "box":
        return f"""
        def Cube "{name}" (
            prepend apiSchemas = [{api}]
        )
        {{{common}
            double size = 1
            double3 xformOp:translate = {_f3(c.center)}
            quatf xformOp:orient = {_f3(c.quat)}
            float3 xformOp:scale = {_f3(c.size)}
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
        }}"""
    pts = np.asarray(c.size)
    from scipy.spatial import ConvexHull  # noqa: PLC0415  (only at asset build time)

    hull = ConvexHull(pts)
    faces = hull.simplices
    # orient faces outward
    ctr = pts.mean(axis=0)
    fixed = []
    for f in faces:
        a, b, cc = pts[f]
        if np.dot(np.cross(b - a, cc - a), a - ctr) < 0:
            f = f[[0, 2, 1]]
        fixed.append(f)
    faces = np.array(fixed)
    return f"""
        def Mesh "{name}" (
            prepend apiSchemas = [{api}, "PhysicsMeshCollisionAPI", "PhysxConvexHullCollisionAPI"]
        )
        {{{common}
            uniform token physics:approximation = "convexHull"
            int physxConvexHullCollision:hullVertexLimit = 64
            int[] faceVertexCounts = [{", ".join("3" for _ in faces)}]
            int[] faceVertexIndices = [{", ".join(str(int(i)) for i in faces.reshape(-1))}]
            point3f[] points = [{", ".join(_f3(p) for p in pts)}]
        }}"""


def _body_usda(b: Body, root: bool) -> str:
    m, com, inertia, axes = b.mass.properties()
    apis = ['"PhysicsRigidBodyAPI"', '"PhysicsMassAPI"', '"PhysxRigidBodyAPI"']
    extra = ""
    if root:
        apis += ['"PhysicsArticulationRootAPI"', '"PhysxArticulationAPI"']
        extra = """
        bool physxArticulation:enabledSelfCollisions = 1
        int physxArticulation:solverPositionIterationCount = 32
        int physxArticulation:solverVelocityIterationCount = 4"""
    if b.name in FILTERED:
        apis.append('"PhysicsFilteredPairsAPI"')
        extra += "\n        rel physics:filteredPairs = [" + ", ".join(f"</{ROOT}/{o}>" for o in FILTERED[b.name]) + "]"
    cols = "".join(_collider_usda(f"col{k}", c) for k, c in enumerate(b.colliders))
    meshes = "".join(uk._mesh_usda(n, mesh, look, ROOT, LOOKS[look], 35.0, indent="            ")
                     for n, mesh, look in b.visuals)
    return f"""
    def Xform "{b.name}" (
        prepend apiSchemas = [{", ".join(apis)}]
    )
    {{
        float physics:mass = {m:.6f}
        point3f physics:centerOfMass = {_f3(com)}
        float3 physics:diagonalInertia = {_f3(np.maximum(inertia, 1e-7))}
        quatf physics:principalAxes = {_f3(axes)}
        float physxRigidBody:maxDepenetrationVelocity = 1
        float physxRigidBody:sleepThreshold = 0
        float physxRigidBody:stabilizationThreshold = 0{extra}
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
{cols}
        def Xform "visual"
        {{{meshes}
        }}
    }}"""


def _joint_usda(j: Joint) -> str:
    lim = ""
    if j.limits is not None:
        lim = f"""
        float physics:lowerLimit = {j.limits[0]:.4f}
        float physics:upperLimit = {j.limits[1]:.4f}"""
    head = ""
    if j.drive:
        head = """ (
        prepend apiSchemas = ["PhysicsDriveAPI:angular"]
    )"""
        lim += """
        uniform token drive:angular:physics:type = "force"
        float drive:angular:physics:stiffness = 0.35
        float drive:angular:physics:damping = 0.01
        float drive:angular:physics:targetPosition = 0"""
    return f"""
    def PhysicsRevoluteJoint "{j.name}"{head}
    {{
        rel physics:body0 = </{ROOT}/{j.body0}>
        rel physics:body1 = </{ROOT}/{j.body1}>
        uniform token physics:axis = "{j.axis}"
        point3f physics:localPos0 = {_f3(j.pos)}
        quatf physics:localRot0 = (1, 0, 0, 0)
        point3f physics:localPos1 = {_f3(j.pos)}
        quatf physics:localRot1 = (1, 0, 0, 0){lim}
    }}"""


def write(path: Path | str = HERE / "assets" / "wheelchair.usda") -> Path:
    path = Path(path)
    bs = bodies()
    looks = "".join(uk._material_usda(n, ROOT, LOOKS[n]) for n in sorted(LOOKS))
    mats = "".join(f"""
        def Material "{n}" (
            prepend apiSchemas = ["PhysicsMaterialAPI", "PhysxMaterialAPI"]
        )
        {{
            float physics:staticFriction = {sf}
            float physics:dynamicFriction = {df}
            float physics:restitution = 0
            uniform token physxMaterial:frictionCombineMode = "average"
        }}""" for n, (sf, df) in PHYSICS_MATERIALS.items())
    text = f"""#usda 1.0
(
    defaultPrim = "{ROOT}"
    metersPerUnit = 1
    kilogramsPerUnit = 1
    upAxis = "Z"
)

def Xform "{ROOT}"
{{
    def Scope "PhysicsMaterials"
    {{{mats}
    }}

    def Scope "Looks"
    {{{looks}
    }}
{"".join(_body_usda(b, i == 0) for i, b in enumerate(bs))}
{"".join(_joint_usda(j) for j in joints())}{"".join(_strap_usda(*st) for st in straps())}
}}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
    return path


# -- geometry the scorer measures ------------------------------------------------------------------------------------

HUB = {side: _v(0.0, s * (W_FOLD + AXLE_OUT), R_WHEEL) for side, s in SIDES.items()}  # in wheel_<side>'s frame
RAIL_ENDS = {side: (_v(X_RAIL[0], s * W_FOLD, Z_RAIL_FOLD), _v(X_RAIL[1], s * W_FOLD, Z_RAIL_FOLD))
             for side, s in SIDES.items()}  # in brace_<side>'s frame
SADDLE_TOPS = {side: [_v(sx, s * W_FOLD, SADDLE_TOP) for sx in SADDLES_X] for side, s in SIDES.items()}  # frame_<side>


def _box_corners(c, size):
    c = _v(*c)
    h = _v(*size) / 2
    return [c + h * _v(i, j, k) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]


def frame_points(side: str) -> np.ndarray:
    """Points on a side frame that touch the floor only if the chair has tipped over (frame_<side>'s frame)."""
    s = SIDES[side]
    y = s * W_FOLD
    pts = [(X_FRONT, y, Z_HOUSING - 0.03), (X_LOWER, y, Z_PIVOT - 0.011), (X_CANE, y, Z_PIVOT - 0.011), (X_CANE, y, HANDLE_END[1]),
           (HANDLE_END[0], y, HANDLE_END[1]), (X_FRONT, y, SADDLE_TOP)]
    pts += _box_corners(((ARM_PAD["x"][0] + ARM_PAD["x"][1]) / 2, y + s * 0.045, 0.74), (0.25, 0.05, 0.04))
    return np.array(pts, float)


def brace_points(side: str) -> np.ndarray:
    s = SIDES[side]
    x = X_MID - s * BRACE_DX
    return np.array([(X_RAIL[0], s * W_FOLD, Z_RAIL_FOLD), (X_RAIL[1], s * W_FOLD, Z_RAIL_FOLD),
                     (x, -s * W_FOLD, Z_PIVOT - 0.016)], float)


def body_samples(name: str) -> np.ndarray:
    """Surface sample points of a body (its own frame), for distances to the fingertips."""
    b = {bb.name: bb for bb in bodies()}[name]
    pts = []
    for c in b.colliders:
        if c.kind == "capsule":
            r, h = c.size
            rot = quat_to_matrix_wxyz(c.quat)
            for t in np.linspace(-h / 2, h / 2, max(2, int(h / 0.02) + 1)):
                for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                    pts.append(c.center + rot @ _v(r * math.cos(a), r * math.sin(a), t))
        elif c.kind == "box":
            rot = quat_to_matrix_wxyz(c.quat)
            n = [max(2, int(sz / 0.02) + 1) for sz in c.size]
            for i in np.linspace(-0.5, 0.5, n[0]):
                for j in np.linspace(-0.5, 0.5, n[1]):
                    for k in np.linspace(-0.5, 0.5, n[2]):
                        if max(abs(i), abs(j), abs(k)) == 0.5:
                            pts.append(c.center + rot @ (_v(i, j, k) * _v(*c.size)))
        else:
            pts.extend(np.asarray(c.size)[::2])
    return np.array(pts, float)


if __name__ == "__main__":
    p = write()
    print(p, f"total mass {TOTAL_MASS:.2f} kg, brace {L_BRACE:.4f} m, folded rail z {Z_RAIL_FOLD:.4f}")
    for b in bodies():
        m, com, inertia, _ = b.mass.properties()
        print(f"{b.name:10s} m={m:.3f} com={np.round(com, 3)} I={np.round(inertia, 4)}")
