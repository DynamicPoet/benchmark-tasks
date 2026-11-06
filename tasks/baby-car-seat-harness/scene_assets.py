"""The scene's models, written as plain-text USD when the task file loads (no pxr needed).

An infant carrier reclined on its base, a newborn doll lying in it, and its three-point harness. The carrier, doll and crotch
buckle are one fixed body. The harness is an articulation with a fixed root:

- each strap is a straight flat band from a slot beside the doll's shoulder to a tongue: two revolute joints at the slot, a
  sliding joint along the band (its two lengths overlap, so it reads as one strap whatever its length), and two revolute joints
  at the tongue, which keep the tongue upright as the strap swings;
- the chest clip is two halves on one carriage. The carriage slides along the doll's chest, and each half slides across it.

Seat coordinates: u up the backrest towards the head, v to the robot's left (the world's y), w out of the backrest. Sizes in
metres; every height is above the table.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import usdkit as uk
from usdkit import Look, Mesh, box, cylinder, join, lathe, prism, tube, write_usda

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"

THETA = math.radians(30.0)  # how far the backrest rises from the table
X0, Z0 = 0.19, 0.06  # the point of the backrest under the doll's crotch
C, S = math.cos(THETA), math.sin(THETA)
R_SEAT = np.array([[C, 0.0, -S], [0.0, 1.0, 0.0], [S, 0.0, C]])  # columns: u, v, w

SHELL_U = (-0.085, 0.30)
SHELL_V = 0.135  # half width, outside
SHELL_WALL = 0.015
SHELL_RIM = 0.050

# The doll
TORSO = ((0.072, 0.018), (0.090, 0.062, 0.028))  # middle (u, w), semi-axes
HEAD = ((0.205, 0.042), 0.052)  # middle (u, w), radius
ARMPIT_U = 0.125

# The harness
SLOT_U, SLOT_V, SLOT_W = 0.155, 0.035, 0.055  # where the straps come out of the backrest
RAIL_W = 0.060  # the clip's middle above the backrest
CLIP_START_U = 0.052  # over the belly
CLIP_TRAVEL = 0.125
HALF = (0.024, 0.024, 0.014)  # u, v, w of each half
HALF_GAP = 0.0135  # each half's inner face from the middle, open

# Tongues
TONGUE = (0.016, 0.022, 0.024)  # plastic body: x, y, z
BLADE = (0.004, 0.018, 0.022)  # metal blade, below the body
# The crotch buckle: its top is level, with two slots facing up
BUCKLE_XY = (0.16, 0.0)
BUCKLE_TOP = 0.088
BUCKLE = (0.040, 0.068, 0.026)
SLOT_OFFSET_Y = 0.016  # each slot's middle from the buckle's middle
SLOT = (0.012, 0.028, 0.020)  # x, y, depth
# Each tongue starts with its blade in a pocket on the cushion, beside the doll's hip
POCKET_XY = (0.216, 0.100)
POCKET_TOP = 0.102
POCKET = (0.036, 0.034)


def seat_point(u: float, v: float, w: float) -> np.ndarray:
    """A point in seat coordinates, in table-top coordinates."""
    return np.array([X0, 0.0, Z0]) + R_SEAT @ np.array([u, v, w])


def to_seat(p: np.ndarray) -> np.ndarray:
    """Table-top coordinates to seat coordinates (rows)."""
    return (np.atleast_2d(p) - np.array([X0, 0.0, Z0])) @ R_SEAT


def tongue_start(side: int) -> np.ndarray:
    return np.array([POCKET_XY[0], side * POCKET_XY[1], POCKET_TOP + TONGUE[2] / 2])


def buckle_slot(side: int) -> np.ndarray:
    """The middle of a slot's mouth, on the buckle's top."""
    return np.array([BUCKLE_XY[0], BUCKLE_XY[1] + side * SLOT_OFFSET_Y, BUCKLE_TOP])


def strap_slot(side: int) -> np.ndarray:
    return seat_point(SLOT_U, side * SLOT_V, SLOT_W)


def clip_start() -> np.ndarray:
    return seat_point(CLIP_START_U, 0.0, RAIL_W)


def _seat_mesh(m: Mesh) -> Mesh:
    """A mesh made in seat coordinates, in table-top coordinates."""
    return Mesh(np.array([X0, 0.0, Z0]) + m.v @ R_SEAT.T, m.f)


def _strap_rotation(side: int) -> np.ndarray:
    d = tongue_start(side) - strap_slot(side)
    x = d / np.linalg.norm(d)
    y = np.cross([0.0, 0.0, 1.0], x)
    y /= np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


def _quat_wxyz(r: np.ndarray) -> np.ndarray:
    tr = r[0, 0] + r[1, 1] + r[2, 2]
    if tr > 0:
        s = 2 * np.sqrt(tr + 1)
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2 * np.sqrt(1 + r[0, 0] - r[1, 1] - r[2, 2])
        q = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif r[1, 1] > r[2, 2]:
        s = 2 * np.sqrt(1 + r[1, 1] - r[0, 0] - r[2, 2])
        q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
    else:
        s = 2 * np.sqrt(1 + r[2, 2] - r[0, 0] - r[1, 1])
        q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def _fmt(v) -> str:
    return "(" + ", ".join(f"{float(a):.7f}" for a in v) + ")"


LOOKS = {
    "shell": Look(color=(0.035, 0.038, 0.045), roughness=0.5),
    "base": Look(color=(0.018, 0.018, 0.02), roughness=0.55),
    "cushion": Look(color=(0.52, 0.56, 0.60), roughness=0.95),
    "handle": Look(color=(0.05, 0.05, 0.055), roughness=0.4),
    "skin": Look(color=(0.86, 0.60, 0.46), roughness=0.55),
    "onesie": Look(color=(0.88, 0.76, 0.34), roughness=0.9),
    "eye": Look(color=(0.02, 0.015, 0.015), roughness=0.2),
    "lip": Look(color=(0.75, 0.35, 0.30), roughness=0.5),
    "webbing": Look(color=(0.015, 0.02, 0.05), roughness=0.85),
    "steel": Look(color=(0.72, 0.73, 0.75), metallic=1.0, roughness=0.3),
    "plastic": Look(color=(0.012, 0.012, 0.014), roughness=0.4),
    "button": Look(color=(0.70, 0.04, 0.04), roughness=0.4),
    "clip": Look(color=(0.72, 0.05, 0.05), roughness=0.35),
    "clip_mark": Look(color=(0.05, 0.05, 0.055), roughness=0.4),
}


def _rings(rings: list[tuple[np.ndarray, float]]) -> Mesh:
    n = len(rings[0][0])
    v = [np.c_[o, np.full(n, z)] for o, z in rings]
    v.append(np.array([[*rings[0][0].mean(axis=0), rings[0][1]], [*rings[-1][0].mean(axis=0), rings[-1][1]]]))
    v = np.vstack(v)
    bottom, top = len(rings) * n, len(rings) * n + 1
    f = []
    for k in range(len(rings) - 1):
        for i in range(n):
            j = (i + 1) % n
            a, b, e, g = k * n + i, k * n + j, (k + 1) * n + j, (k + 1) * n + i
            f += [(a, b, e), (a, e, g)]
    last = (len(rings) - 1) * n
    for i in range(n):
        j = (i + 1) % n
        f += [(bottom, j, i), (top, last + i, last + j)]
    return Mesh(v, np.array(f))


def _ellipsoid(centre, semi, segments: int = 24, rings: int = 12) -> Mesh:
    prof = [(math.sin(a), -math.cos(a)) for a in np.linspace(0.0, math.pi, rings + 1)]
    m = lathe([(0.0 if abs(r) < 1e-6 else r, z) for r, z in prof], segments)
    return Mesh(m.v * np.asarray(semi, float) + np.asarray(centre, float), m.f)


def _collider_at(name: str, size, centre, rot: np.ndarray | None, root: str) -> str:
    q = _quat_wxyz(np.eye(3) if rot is None else rot)
    return f"""
    def Cube "{name}" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
    )
    {{
        rel material:binding:physics = </{root}/PhysicsMaterial>
        uniform token purpose = "guide"
        double size = 1
        float physxCollision:contactOffset = 0.004
        float physxCollision:restOffset = 0
        double3 xformOp:translate = {_fmt(centre)}
        quatd xformOp:orient = {_fmt(q)}
        float3 xformOp:scale = {_fmt(size)}
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    }}"""


def _sphere_collider(name: str, radius: float, centre, root: str) -> str:
    return f"""
    def Sphere "{name}" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
    )
    {{
        rel material:binding:physics = </{root}/PhysicsMaterial>
        uniform token purpose = "guide"
        double radius = {radius:.6f}
        float physxCollision:contactOffset = 0.004
        float physxCollision:restOffset = 0
        double3 xformOp:translate = {_fmt(centre)}
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}"""


def _slotted_block(name: str, centre_xy, top: float, size_xy, height: float, slots: list[float], slot, root: str) -> list[str]:
    """A block with slots along its x middle, as boxes: a floor, two long sides, and walls between the slots (y offsets)."""
    cx, cy = centre_xy
    sx, sy = size_xy
    sl_x, sl_y, depth = slot
    out = [_collider_at(f"{name}_floor", (sx, sy, height - depth), (cx, cy, top - depth - (height - depth) / 2), None, root)]
    side = (sx - sl_x) / 2
    for k, s in enumerate((-1, 1)):
        out.append(_collider_at(f"{name}_side{k}", (side, sy, depth), (cx + s * (sl_x + side) / 2, cy, top - depth / 2), None, root))
    edges = [-sy / 2] + [e for off in slots for e in (off - sl_y / 2, off + sl_y / 2)] + [sy / 2]
    for k in range(0, len(edges), 2):
        y0, y1 = edges[k], edges[k + 1]
        if y1 - y0 > 1e-6:
            out.append(_collider_at(f"{name}_wall{k}", (sl_x, y1 - y0, depth), (cx, cy + (y0 + y1) / 2, top - depth / 2), None, root))
    return out


def _doll() -> list[tuple[str, Mesh, str]]:
    (tu, tw), semi = TORSO
    torso = _ellipsoid((tu, 0.0, tw), semi)
    (hu, hw), r = HEAD
    head = _ellipsoid((hu, 0.0, hw), (r, r * 0.92, r))
    neck = _ellipsoid((0.156, 0.0, 0.026), (0.022, 0.030, 0.020), 16, 8)
    parts = [("torso", torso, "onesie"), ("head", head, "skin"), ("neck", neck, "skin")]
    sleeves, hands, thighs, feet = [], [], [], []
    for s in (-1, 1):
        sleeves.append(tube([(0.135, s * 0.058, 0.024), (0.105, s * 0.070, 0.022), (0.065, s * 0.078, 0.020), (0.042, s * 0.078, 0.022)], 0.017, 12))
        sleeves.append(_ellipsoid((0.135, s * 0.058, 0.024), (0.022, 0.020, 0.018), 16, 8))
        hands.append(_ellipsoid((0.030, s * 0.074, 0.026), (0.018, 0.015, 0.012), 16, 8))
        thighs.append(tube([(0.015, s * 0.046, 0.022), (-0.02, s * 0.056, 0.034), (-0.052, s * 0.066, 0.046), (-0.070, s * 0.066, 0.034), (-0.072, s * 0.062, 0.020)], 0.019, 12))
        feet.append(_ellipsoid((-0.062, s * 0.056, 0.016), (0.026, 0.017, 0.015), 16, 8))
    parts += [("sleeves", join(*sleeves), "onesie"), ("hands", join(*hands), "skin"), ("legs", join(*thighs), "onesie"), ("feet", join(*feet), "skin")]
    # A face looking up and towards the robot: eyes, a nose, a mouth, on the head's surface
    look = np.array([-0.55, 0.0, 0.83])
    side = np.array([0.0, 1.0, 0.0])
    up = np.cross(side, look)
    centre = np.array([hu, 0.0, hw])

    def on_head(dx: float, dy: float) -> np.ndarray:
        d = look + dx * up + dy * side
        d /= np.linalg.norm(d)
        return centre + d * np.array([r, r * 0.92, r])

    eyes = join(*(_ellipsoid(on_head(0.18, s * 0.34), (0.0068, 0.0068, 0.0040), 12, 6) for s in (-1, 1)))
    nose = _ellipsoid(on_head(-0.08, 0.0), (0.0065, 0.0075, 0.0055), 12, 6)
    mouth = _ellipsoid(on_head(-0.38, 0.0), (0.0040, 0.012, 0.0030), 12, 6)
    parts += [("eyes", eyes, "eye"), ("nose", nose, "skin"), ("mouth", mouth, "lip")]
    return parts


def write_seat() -> str:
    root = "Seat"
    su = SHELL_U[1] - SHELL_U[0]
    sv = 2 * SHELL_V
    uc = (SHELL_U[0] + SHELL_U[1]) / 2
    o = uk._rounded_outline
    seg = 6
    wall = SHELL_WALL
    shell = _rings([
        (o(su - 0.05, sv - 0.05, 0.05, seg), -0.03),
        (o(su, sv, 0.07, seg), -0.002),
        (o(su, sv, 0.07, seg), SHELL_RIM - 0.006),
        (o(su - 0.008, sv - 0.008, 0.066, seg), SHELL_RIM),
        (o(su - 2 * wall + 0.008, sv - 2 * wall + 0.008, 0.052, seg), SHELL_RIM),
        (o(su - 2 * wall, sv - 2 * wall, 0.050, seg), SHELL_RIM - 0.006),
        (o(su - 2 * wall - 0.004, sv - 2 * wall - 0.004, 0.048, seg), 0.012),
        (o(su - 2 * wall - 0.030, sv - 2 * wall - 0.030, 0.035, seg), 0.0),
    ]).moved((uc, 0.0, 0.0))
    cushion = prism(o(su - 2 * wall - 0.012, sv - 2 * wall - 0.012, 0.040, seg), 0.0, 0.008, 0.003).moved((uc, 0.0, 0.0))
    base = prism(np.array([(0.155, 0.0), (0.45, 0.0), (0.45, 0.145), (0.40, 0.141)]), -0.125, 0.125, 0.005)
    base = base.turned(90.0, "x")
    # A carry handle, folded back behind the head
    arc = [(0.135, 0.127, 0.040), (0.20, 0.13, 0.040), (0.28, 0.13, 0.040), (0.345, 0.11, 0.045), (0.37, 0.05, 0.05), (0.375, 0.0, 0.05),
           (0.37, -0.05, 0.05), (0.345, -0.11, 0.045), (0.28, -0.13, 0.040), (0.20, -0.13, 0.040), (0.135, -0.127, 0.040)]
    handle = tube(arc, 0.011, 12)
    hubs = join(*(cylinder(0.022, 0.012, center=(0.135, s * 0.133, 0.040), axis="y") for s in (-1, 1)))
    parts: list[tuple] = [("shell", _seat_mesh(shell), "shell"), ("cushion", _seat_mesh(cushion), "cushion"), ("base", base, "base"),
                          ("handle", _seat_mesh(handle), "handle"), ("hubs", _seat_mesh(hubs), "handle")]
    parts += [(n, _seat_mesh(m), lk) for n, m, lk in _doll()]

    bx, by = BUCKLE_XY
    bs = BUCKLE
    body = box((bs[0], bs[1], bs[2]), center=(bx, by, BUCKLE_TOP - bs[2] / 2), bevel=0.003, round_xy=0.006)
    stalk = box((0.022, 0.022, 0.05), center=(bx + 0.006, by, BUCKLE_TOP - bs[2] - 0.022))
    mouths = join(*(box((SLOT[0], SLOT[1] - 0.002, 0.0008), center=(bx, by + s * SLOT_OFFSET_Y, BUCKLE_TOP + 0.0002)) for s in (-1, 1)))
    button = box((0.002, 0.026, 0.012), center=(bx - bs[0] / 2 - 0.0006, by, BUCKLE_TOP - 0.012), bevel=0.0005)
    parts += [("buckle", body, "plastic"), ("stalk", stalk, "webbing"), ("mouths", mouths, "eye"), ("button", button, "button")]

    colliders: list[str] = []
    for s in (-1, 1):
        px, py = POCKET_XY[0], s * POCKET_XY[1]
        height = POCKET_TOP - 0.05
        pocket = box((POCKET[0], POCKET[1], height), center=(px, py, POCKET_TOP - height / 2), bevel=0.003, round_xy=0.005)
        parts.append((f"pocket{'LR'[s > 0]}", pocket, "shell"))
        colliders += _slotted_block(f"pocket{'LR'[s > 0]}", (px, py), POCKET_TOP, POCKET, height, [0.0], (SLOT[0], SLOT[1], SLOT[2]), root)
    colliders += _slotted_block("buckle", BUCKLE_XY, BUCKLE_TOP, (bs[0], bs[1]), bs[2], [-SLOT_OFFSET_Y, SLOT_OFFSET_Y], SLOT, root)

    def slanted(name, size, uvw) -> str:
        return _collider_at(name, size, seat_point(*uvw), R_SEAT, root)

    inner_v = SHELL_V - wall
    colliders += [
        slanted("back", (su, sv, 0.03), (uc, 0.0, -0.015)),
        slanted("side_l", (su, wall, SHELL_RIM), (uc, inner_v + wall / 2, SHELL_RIM / 2)),
        slanted("side_r", (su, wall, SHELL_RIM), (uc, -inner_v - wall / 2, SHELL_RIM / 2)),
        slanted("foot", (wall, sv, SHELL_RIM), (SHELL_U[0] + wall / 2, 0.0, SHELL_RIM / 2)),
        slanted("top_end", (wall, sv, SHELL_RIM), (SHELL_U[1] - wall / 2, 0.0, SHELL_RIM / 2)),
        slanted("torso", (0.125, 0.11, 0.040), (0.0925, 0.0, 0.020)),  # from above the buckle to the shoulders
        _sphere_collider("head", HEAD[1] - 0.002, seat_point(HEAD[0][0], 0.0, HEAD[0][1]), root),
    ]
    return write_usda(str(ASSETS / "seat.usda"), root, parts, LOOKS, colliders, mass=1.0, friction=(0.6, 0.5), crease=55.0)


def _tongue_meshes(r: np.ndarray) -> list[tuple[str, Mesh, str]]:
    """The tongue, upright in the world, in a link frame rotated by r."""
    tx, ty, tz = TONGUE
    bx, by, bz = BLADE
    body = box((tx, ty, tz), bevel=0.002, round_xy=0.003)
    blade = box((bx, by, bz), center=(0.0, 0.0, -tz / 2 - bz / 2))
    return [("body", Mesh(body.v @ r, body.f), "plastic"), ("blade", Mesh(blade.v @ r, blade.f), "steel")]


def write_harness() -> str:
    root = "Harness"
    used = {"webbing", "plastic", "steel", "clip", "clip_mark"}
    looks = "".join(uk._material_usda(k, root, LOOKS[k]) for k in sorted(used))
    links: list[str] = []
    joints: list[str] = []

    def link(name: str, pos, rot: np.ndarray, mass: float, inertia: float, body: str = "", gravity: bool = False) -> str:
        grav = "" if gravity else "\n        bool physxRigidBody:disableGravity = 1"
        return f"""
    def Xform "{name}" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
    )
    {{
        float physics:mass = {mass:.5f}
        float3 physics:diagonalInertia = ({inertia:.3e}, {inertia:.3e}, {inertia:.3e})
        point3f physics:centerOfMass = (0, 0, 0)
        float physxRigidBody:maxDepenetrationVelocity = 1{grav}
        double3 xformOp:translate = {_fmt(pos)}
        quatd xformOp:orient = {_fmt(_quat_wxyz(rot))}
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
{body}
    }}
"""

    def joint(kind: str, name: str, b0: str, b1: str, pos0, rot0, axis: str, lo: float, hi: float) -> str:
        drive = "linear" if kind == "Prismatic" else "angular"
        return f"""
    def Physics{kind}Joint "{name}" (
        prepend apiSchemas = ["PhysicsDriveAPI:{drive}"]
    )
    {{
        uniform token drive:{drive}:physics:type = "force"
        float drive:{drive}:physics:stiffness = 0
        float drive:{drive}:physics:damping = 0
        rel physics:body0 = </{root}/{b0}>
        rel physics:body1 = </{root}/{b1}>
        uniform token physics:axis = "{axis}"
        point3f physics:localPos0 = {_fmt(pos0)}
        quatf physics:localRot0 = {_fmt(_quat_wxyz(rot0))}
        point3f physics:localPos1 = (0, 0, 0)
        quatf physics:localRot1 = (1, 0, 0, 0)
        float physics:lowerLimit = {lo:.5f}
        float physics:upperLimit = {hi:.5f}
    }}
"""

    def mesh(p, indent="        ") -> str:
        return uk._mesh_usda(*p, root, LOOKS[p[2]], 40.0, indent=indent)

    def cube(name: str, size, centre, rot: np.ndarray | None = None, patch: float = 0.0) -> str:
        q = _quat_wxyz(np.eye(3) if rot is None else rot)
        extra = f"\n            float physxCollision:torsionalPatchRadius = {patch:.4f}\n            float physxCollision:minTorsionalPatchRadius = {0.8 * patch:.4f}" if patch else ""
        return f"""
        def Cube "{name}" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
        )
        {{
            rel material:binding:physics = </{root}/PhysicsMaterial>
            uniform token purpose = "guide"
            double size = 1
            float physxCollision:contactOffset = 0.003
            float physxCollision:restOffset = 0{extra}
            double3 xformOp:translate = {_fmt(centre)}
            quatd xformOp:orient = {_fmt(q)}
            float3 xformOp:scale = {_fmt(size)}
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
        }}"""

    links.append(link("frame", (0, 0, 0), np.eye(3), 1.0, 1e-3))
    for side, tag in ((1, "l"), (-1, "r")):
        slot, t0 = strap_slot(side), tongue_start(side)
        r = _strap_rotation(side)
        length = float(np.linalg.norm(t0 - slot))
        band = 0.11
        near = box((band, 0.022, 0.002), center=(band / 2, 0.0, 0.0))
        far = box((band, 0.022, 0.002), center=(-band / 2, 0.0, 0.0005))
        links.append(link(f"yaw_{tag}", slot, r, 0.004, 1e-6))
        links.append(link(f"pitch_{tag}", slot, r, 0.006, 2e-6, mesh(("band", near, "webbing"))))
        links.append(link(f"stage_{tag}", t0, r, 0.006, 2e-6, mesh(("band", far, "webbing"))))
        links.append(link(f"bend_{tag}", t0, r, 0.004, 1e-6))
        rt = r.T
        tz = TONGUE[2]
        collide = cube("body_c", TONGUE, (0, 0, 0), rt, 0.008) + cube("blade_c", BLADE, rt @ np.array([0, 0, -tz / 2 - BLADE[2] / 2]), rt)
        body = collide + "".join(mesh(p) for p in _tongue_meshes(r))
        links.append(link(f"tongue_{tag}", t0, r, 0.025, 4e-6, body, gravity=True))
        joints.append(joint("Revolute", f"strap_yaw_{tag}", "frame", f"yaw_{tag}", slot, r, "Z", -120.0, 120.0))
        joints.append(joint("Revolute", f"strap_pitch_{tag}", f"yaw_{tag}", f"pitch_{tag}", (0, 0, 0), np.eye(3), "Y", -120.0, 120.0))
        joints.append(joint("Prismatic", f"strap_len_{tag}", f"pitch_{tag}", f"stage_{tag}", (length, 0, 0), np.eye(3), "X", -0.09, 0.12))
        joints.append(joint("Revolute", f"tongue_pitch_{tag}", f"stage_{tag}", f"bend_{tag}", (0, 0, 0), np.eye(3), "Y", -120.0, 120.0))
        joints.append(joint("Revolute", f"tongue_yaw_{tag}", f"bend_{tag}", f"tongue_{tag}", (0, 0, 0), np.eye(3), "Z", -120.0, 120.0))

    start = clip_start()
    links.append(link("carriage", start, R_SEAT, 0.02, 1e-5))
    joints.append(joint("Prismatic", "clip_slide", "frame", "carriage", start, R_SEAT, "X", -0.012, CLIP_TRAVEL))
    hu, hv, hw = HALF
    for tag, s in (("l", 1), ("r", -1)):
        centre = (0.0, s * hv / 2, 0.0)
        shape = box(HALF, center=centre, bevel=0.002, round_xy=0.004)
        mark = box((0.010, 0.012, 0.0008), center=(0.0, s * hv * 0.62, hw / 2 + 0.0002))
        body = cube("half_c", HALF, centre, None, 0.008) + mesh(("half", shape, "clip")) + mesh(("mark", mark, "clip_mark"))
        links.append(link(f"half_{tag}", start + R_SEAT @ np.array([0.0, s * HALF_GAP, 0.0]), R_SEAT, 0.015, 3e-6, body))
        lo, hi = (-HALF_GAP, 0.0) if s > 0 else (0.0, HALF_GAP)
        joints.append(joint("Prismatic", f"clip_half_{tag}", "carriage", f"half_{tag}", (0.0, s * HALF_GAP, 0.0), np.eye(3), "Y", lo, hi))
    text = f"""#usda 1.0
(
    defaultPrim = "{root}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}" (
    prepend apiSchemas = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"]
)
{{
    bool physxArticulation:enabledSelfCollisions = 0
    int physxArticulation:solverPositionIterationCount = 32
    int physxArticulation:solverVelocityIterationCount = 4

    def Material "PhysicsMaterial" (
        prepend apiSchemas = ["PhysicsMaterialAPI"]
    )
    {{
        float physics:staticFriction = 0.8
        float physics:dynamicFriction = 0.7
        float physics:restitution = 0
    }}

    def Scope "Looks"
    {{{looks}
    }}
{''.join(links)}
    def PhysicsFixedJoint "world_joint"
    {{
        rel physics:body1 = </{root}/frame>
    }}
{''.join(joints)}}}
"""
    os.makedirs(ASSETS, exist_ok=True)
    path = ASSETS / "harness.usda"
    path.write_text(text)
    return str(path)


def write_all() -> None:
    write_seat()
    write_harness()
