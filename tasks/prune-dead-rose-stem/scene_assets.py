"""The scene's models, written as plain-text USD when the task file loads (no pxr needed).

- **The rose bush** (``bush.usda``): a terracotta pot of soil with four green, leafy, thorny canes and one dead cane at its front,
  as one articulation whose root is the pot (free on the table). The dead cane is green from the soil up to just above a red bud
  and brown above, up to an old pruning cut. The first cut frees its upper section at any height the blades reach; later cuts
  can shorten the retained stub and remove its original bud:
  - the stub left in the pot is two sleeves, green and brown, which slide down into the soil to the height of the cut;
  - the part cut off is a short top piece with two sleeves inside it, which slide out down to the height of the cut. Until the
    cut, a chain of six stiffly driven joints holds that part in place on the pot; the cut releases them, and from then on it moves
    as a free body.
  Every slide is driven stiffly, so each part stays rigid. The nested sleeves never collide with one another; the cut-off part
  collides with the pot, the soil and the green canes.
- **Bypass secateurs** (``secateurs.usda``): two halves on a pivot with a spring that holds them open. Squeezing the handles closes
  the blade past the hook. The handles and the pivot collide; the blade and the hook do not (the task file judges the cut).
- **The bin** (``bin.usda``): an open black plastic garden tub, free on the table.

Sizes in metres. The station frame: x forward, y left, heights above the table top.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

import usdkit as uk
from usdkit import Look, Mesh, box, join, lathe, prism

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
RNG_SEED = 7

# -- layout -----------------------------------------------------------------------------------------------------------

POT_XY = (0.37, 0.03)
POT_R_BOTTOM, POT_R_TOP, POT_H = 0.085, 0.110, 0.075
SOIL_Z = 0.066
CANE_BASE = (0.33, 0.0)  # where the dead cane leaves the soil, at the front of the pot
CANE_IN_POT = np.array([CANE_BASE[0] - POT_XY[0], CANE_BASE[1] - POT_XY[1], SOIL_Z])
LEAN = math.radians(55.0)  # the dead cane leans out over the pot's rim, toward the robot's left
CANE_DIR = np.array([0.0, math.sin(LEAN), math.cos(LEAN)])

R0 = 0.0050   # the dead cane's radius: the stub's green sleeve and the cutting's top piece
R1 = 0.00485  # its brown sleeves
R2 = 0.0047   # the cutting's green sleeve
# Distances along the dead cane, from where it leaves the soil:
BUD_S = 0.055       # the bud's middle
BUD_TOP_S = 0.059   # its top
BROWN_S = BUD_TOP_S + 0.020  # where the brown, dead wood starts
TOP_S = 0.240       # the old pruning cut at the cane's tip
T_LOW_S = 0.155     # the cutting's top piece starts here; a cut further out parts here
H_MIN_S = 0.045     # the lowest cut the parts can show (the pot's rim keeps the blades above it)

SG_BOTTOM_S = -0.010
L_SG = BROWN_S - SG_BOTTOM_S        # stub, green sleeve
L_SB = T_LOW_S - BROWN_S + 0.002    # stub, brown sleeve: its end 2 mm inside the top piece
L_T = TOP_S - T_LOW_S               # cutting, top piece
CB_HIDE, CG_HIDE = 0.002, 0.003     # where the cutting's sleeves sit inside the top piece before the cut
L_CB = T_LOW_S - BROWN_S + 0.002    # cutting, brown sleeve
L_CG = BROWN_S - H_MIN_S + 0.002    # cutting, green sleeve
BUD_SINK = -0.05                    # the stub's bud sinks this far (straight down, into the soil) when the cut takes it
BUD_HIDDEN = (R0 - 0.0015, 0.0, 0.030)  # the cutting's bud, hidden inside the top piece (cane frame)
BUD_SHOWN = (0.0, 0.0, BUD_S - T_LOW_S)  # ... and shown on the cutting, where it grew
BUD_TRAVEL = float(np.linalg.norm(np.subtract(BUD_SHOWN, BUD_HIDDEN)))


def on_cane(s: float) -> np.ndarray:
    """A point on the dead cane's axis, s along it from the soil, in the pot frame."""
    return CANE_IN_POT + s * CANE_DIR


# Green canes, in the pot frame: (base on the soil, top, radius). They stand behind the dead cane and lean a little outward.
GREEN_CANES = [
    ((0.060, -0.040), (0.085, -0.060, 0.300), 0.0042),
    ((0.070, 0.030), (0.100, 0.040, 0.320), 0.0045),
    ((0.035, 0.075), (0.050, 0.095, 0.290), 0.0040),
    ((0.055, 0.060), (0.080, 0.075, 0.270), 0.0042),
]
GREEN_CUT_FROM = SOIL_Z + 0.01  # a green cane counts as cut if the blades close across it above this

# Bypass secateurs, in the tool frame: pivot at the origin, the jaws pointing +y, z up through the tool's middle.
JAW_OPEN = 0.62     # rad between the blade's and the hook's edges at rest
OVERLAP = 0.06      # rad the blade passes the hook when the handles are fully squeezed
CLOSE = JAW_OPEN + OVERLAP  # the pivot's travel
HALF = CLOSE / 2    # each half turns this far between closed and open
HANDLE_X = 0.012    # when closed, the handles' middles lie this far either side of the axis
HANDLE_W, HANDLE_T = 0.016, 0.018
GRIP_Y = (-0.036, -0.130)  # the red grips along each handle (closed)
BLADE_Z = 0.0018    # the blade's middle, above the tool's middle: the cutting plane
JAW_REACH = (0.008, 0.055)  # a cane between the blade and the hook, this far from the pivot, is in the jaws
TOOL_HALF_T = 0.0095  # the tool's lowest point below its middle
SECATEURS_START = (0.22, -0.13)  # the pivot on the table at the start; the jaws point left (+y)
SPRING_K = 1.0      # N m/rad
SPRING_PRELOAD = 0.30  # rad: the spring pushes the halves against their open stop with 0.3 N m

# The bin: an open black plastic garden tub.
BIN_XY = (0.20, 0.37)
BIN_R_BOTTOM, BIN_R_TOP, BIN_H, BIN_WALL = 0.110, 0.130, 0.120, 0.004

# -- looks ------------------------------------------------------------------------------------------------------------

LOOKS = {
    "terracotta": Look(color=(0.52, 0.17, 0.075), roughness=0.85),
    "terracotta_dark": Look(color=(0.36, 0.11, 0.05), roughness=0.9),
    "soil": Look(color=(0.050, 0.028, 0.014), roughness=1.0),
    "bark": Look(color=(0.12, 0.075, 0.04), roughness=0.95),
    "grit": Look(color=(0.30, 0.27, 0.22), roughness=0.9),
    "cane": Look(color=(0.075, 0.17, 0.025), roughness=0.55),
    "cane_young": Look(color=(0.16, 0.12, 0.02), roughness=0.55),
    "dead": Look(color=(0.16, 0.085, 0.040), roughness=0.9),
    "dead_dark": Look(color=(0.075, 0.035, 0.018), roughness=0.9),
    "pith": Look(color=(0.78, 0.74, 0.50), roughness=0.8),
    "pith_dead": Look(color=(0.28, 0.17, 0.08), roughness=0.9),
    "old_cut": Look(color=(0.10, 0.07, 0.05), roughness=0.95),
    "bud": Look(color=(0.45, 0.02, 0.012), roughness=0.5),
    "thorn": Look(color=(0.30, 0.07, 0.03), roughness=0.5),
    "leaf": Look(color=(0.022, 0.085, 0.012), roughness=0.42),
    "leaf_young": Look(color=(0.10, 0.07, 0.012), roughness=0.45),
    "petal": Look(color=(0.72, 0.035, 0.11), roughness=0.55),
    "petal_dark": Look(color=(0.45, 0.012, 0.06), roughness=0.55),
    "sepal": Look(color=(0.03, 0.10, 0.015), roughness=0.5),
    "grip": Look(color=(0.60, 0.010, 0.008), roughness=0.45),
    "steel": Look(color=(0.78, 0.78, 0.80), metallic=1.0, roughness=0.22),
    "steel_dark": Look(color=(0.30, 0.30, 0.32), metallic=1.0, roughness=0.35),
    "anvil": Look(color=(0.035, 0.035, 0.040), metallic=0.6, roughness=0.45),
    "brass": Look(color=(0.70, 0.52, 0.18), metallic=1.0, roughness=0.3),
    "bin": Look(color=(0.018, 0.018, 0.020), roughness=0.65),
}

# -- small geometry helpers -------------------------------------------------------------------------------------------


def rod(z0: float, z1: float, r: float, sides: int = 18) -> Mesh:
    """An open tube along z (no end caps)."""
    return lathe([(r, z0), (r, z1)], sides)


def cap(z: float, r: float, up: bool, sides: int = 18) -> Mesh:
    """A flat disc at height z, facing up or down."""
    return lathe([(r, z), (0.0, z)] if up else [(0.0, z), (r, z)], sides)


def _frame(d: np.ndarray) -> np.ndarray:
    """A rotation whose z axis is d."""
    d = d / np.linalg.norm(d)
    a = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(a, d)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(d, x), d], axis=1)


def placed(m: Mesh, at, direction) -> Mesh:
    """A mesh built along +z, turned so that +z points along `direction` and moved to `at`."""
    rot = _frame(np.asarray(direction, float))
    return Mesh(m.v @ rot.T + np.asarray(at, float), m.f)


def spun(m: Mesh, angle: float) -> Mesh:
    """Turned about z by `angle` (rad)."""
    c, s = math.cos(angle), math.sin(angle)
    return Mesh(m.v @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]).T, m.f)


def thorn(at, outward, up: float = 0.35, size: float = 1.0) -> Mesh:
    d = np.asarray(outward, float)
    d = d / np.linalg.norm(d) + np.array([0.0, 0.0, up])
    cone = lathe([(0.0, -0.001 * size), (0.0016 * size, -0.001 * size), (0.0006 * size, 0.0025 * size), (0.0, 0.0045 * size)], 8)
    return placed(cone, at, d)


def leaflet(length: float, width: float, serrate: int = 9) -> Mesh:
    """A rose leaflet: a pointed oval with a toothed edge, 0.6 mm thick, along +x from its base."""
    pts = []
    n = 4 * serrate
    for k in range(n):
        t = 2 * math.pi * k / n
        x = 0.5 * length * (1 - math.cos(t)) if t <= math.pi else 0.5 * length * (1 - math.cos(t))
        y = 0.5 * width * math.sin(t) * (1.0 - 0.25 * (x / length) ** 2)
        tooth = 0.0012 if (k % 4 == 1 and 0.15 < x / length < 0.95) else 0.0
        pts.append((x, y + math.copysign(tooth, y)))
    outline = np.array(pts)
    outline = outline[::-1] if np.cross(outline[1] - outline[0], outline[2] - outline[1]) < 0 else outline
    return prism(outline, -0.0003, 0.0003)


def compound_leaf(at, direction: float, droop: float, scale: float, rng) -> Mesh:
    """A rose leaf: a stalk with two pairs of leaflets and one at the tip, leaving the cane at `at` toward `direction` (rad)."""
    stalk_len = 0.055 * scale
    parts = [placed(lathe([(0.0009, 0.0), (0.0006, stalk_len)], 6), (0, 0, 0), (1.0, 0.0, 0.0))]
    for k, f in enumerate((0.45, 0.75)):
        for side in (-1, 1):
            leaf = leaflet(0.026 * scale, 0.016 * scale)
            leaf = spun(leaf, side * math.radians(62 - 8 * k))
            leaf = Mesh(leaf.v @ _rx(side * 0.25).T, leaf.f)
            parts.append(leaf.moved((stalk_len * f, 0.0, 0.0)))
    parts.append(leaflet(0.032 * scale, 0.019 * scale).moved((stalk_len * 0.98, 0.0, 0.0)))
    m = join(*parts)
    m = Mesh(m.v @ _ry(droop).T, m.f)
    m = spun(m, direction + rng.uniform(-0.15, 0.15))
    return m.moved(at)


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rose_flower(at, scale: float = 1.0) -> list[tuple[Mesh, str]]:
    """A half-open red rose: rings of cupped petals round a tight centre, on green sepals."""
    parts: list[tuple[Mesh, str]] = []
    for ring, (n, r, tilt, size) in enumerate(((5, 0.016, 1.05, 0.024), (5, 0.012, 0.75, 0.021), (4, 0.008, 0.45, 0.017), (3, 0.004, 0.2, 0.012))):
        for k in range(n):
            a = 2 * math.pi * (k + 0.5 * ring) / n
            petal = lathe([(0.0, 0.0), (size * 0.32, size * 0.05), (size * 0.40, size * 0.45), (size * 0.30, size * 0.85), (0.0, size)], 10)
            petal = petal.scaled((1.0, 0.35, 1.0))
            petal = Mesh(petal.v @ _rx(-tilt).T, petal.f)
            petal = spun(petal, a + math.pi / 2).moved((r * math.cos(a), r * math.sin(a), 0.0))
            parts.append((petal.scaled(scale).moved(at), "petal" if ring < 2 else "petal_dark"))
    for k in range(5):
        a = 2 * math.pi * k / 5 + 0.3
        sepal = lathe([(0.0, 0.0), (0.004, 0.003), (0.0015, 0.018), (0.0, 0.022)], 6).scaled((1.0, 0.3, 1.0))
        sepal = spun(Mesh(sepal.v @ _rx(-2.0).T, sepal.f), a + math.pi / 2)
        parts.append((sepal.scaled(scale).moved(at), "sepal"))
    return parts


# -- USD text helpers -------------------------------------------------------------------------------------------------


def _q_wxyz_z(angle: float) -> tuple[float, float, float, float]:
    return (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))


def _q_wxyz_from_matrix(m: np.ndarray) -> tuple[float, float, float, float]:
    w = math.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) / 2, m[2, 1] - m[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - m[0, 0] + m[1, 1] - m[2, 2])) / 2, m[0, 2] - m[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - m[0, 0] - m[1, 1] + m[2, 2])) / 2, m[1, 0] - m[0, 1])
    return (w, x, y, z)


def _f3(v) -> str:
    return f"({v[0]:.7f}, {v[1]:.7f}, {v[2]:.7f})"


def _f4(q) -> str:
    return f"({q[0]:.7f}, {q[1]:.7f}, {q[2]:.7f}, {q[3]:.7f})"


def obox(name: str, size, center, root: str, quat=(1.0, 0.0, 0.0, 0.0), patch: float = 0.0, indent: str = "        ") -> str:
    """A box collider, not rendered, turned by `quat` (w, x, y, z)."""
    extra = (f"\n{indent}    float physxCollision:torsionalPatchRadius = {patch:.4f}"
             f"\n{indent}    float physxCollision:minTorsionalPatchRadius = {0.8 * patch:.4f}") if patch else ""
    return f"""
{indent}def Cube "{name}" (
{indent}    prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
{indent})
{indent}{{
{indent}    rel material:binding:physics = </{root}/PhysicsMaterial>
{indent}    uniform token purpose = "guide"
{indent}    double size = 1
{indent}    float physxCollision:contactOffset = 0.003
{indent}    float physxCollision:restOffset = 0{extra}
{indent}    double3 xformOp:translate = {_f3(center)}
{indent}    quatd xformOp:orient = {_f4(quat)}
{indent}    float3 xformOp:scale = {_f3(size)}
{indent}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
{indent}}}"""


def segment_box(name: str, p0, p1, radius: float, root: str, indent: str = "        ") -> str:
    """A square collider round the segment p0-p1."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    q = _q_wxyz_from_matrix(_frame(d))
    return obox(name, (2 * radius, 2 * radius, float(np.linalg.norm(d))), (p0 + p1) / 2, root, q, indent=indent)


def hull(name: str, mesh: Mesh, root: str, indent: str = "        ") -> str:
    """A convex-hull collider from a mesh, not rendered."""
    return f"""
{indent}def Mesh "{name}" (
{indent}    prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "PhysicsMeshCollisionAPI", "MaterialBindingAPI"]
{indent})
{indent}{{
{indent}    uniform token physics:approximation = "convexHull"
{indent}    rel material:binding:physics = </{root}/PhysicsMaterial>
{indent}    uniform token purpose = "guide"
{indent}    float physxCollision:contactOffset = 0.003
{indent}    float physxCollision:restOffset = 0
{indent}    int[] faceVertexCounts = [{", ".join("3" for _ in range(len(mesh.f)))}]
{indent}    int[] faceVertexIndices = [{", ".join(str(int(i)) for i in mesh.f.reshape(-1))}]
{indent}    point3f[] points = [{uk._fmt(mesh.v)}]
{indent}}}"""


def visuals(parts, root: str, indent: str = "        ") -> str:
    return "".join(uk._mesh_usda(name, mesh, look, root, LOOKS[look], 35.0, indent=indent) for name, mesh, look in parts)


def link(name: str, mass: float, inertia, body: str, at=(0.0, 0.0, 0.0), gravity: bool = True, com=(0.0, 0.0, 0.0),
         extra_api: str = "", extra: str = "", orient=(1.0, 0.0, 0.0, 0.0)) -> str:
    return f"""
    def Xform "{name}" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"{extra_api}]
    )
    {{
        float physics:mass = {mass:.6f}
        float3 physics:diagonalInertia = ({inertia[0]:.3e}, {inertia[1]:.3e}, {inertia[2]:.3e})
        point3f physics:centerOfMass = {_f3(com)}
        bool physxRigidBody:disableGravity = {0 if gravity else 1}
        float physxRigidBody:maxDepenetrationVelocity = 1
        double3 xformOp:translate = {_f3(at)}
        quatd xformOp:orient = {_f4(orient)}
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]{extra}
{body}
    }}
"""


def joint(kind: str, name: str, root: str, b0: str, b1: str, axis: str = "Z", pos0=(0.0, 0.0, 0.0), rot0=(1.0, 0.0, 0.0, 0.0),
          pos1=(0.0, 0.0, 0.0), rot1=(1.0, 0.0, 0.0, 0.0), limits: tuple[float, float] | None = None) -> str:
    """A prismatic or revolute joint with a drive (its gains are set by the task file). Revolute limits in degrees."""
    api = "linear" if kind == "Prismatic" else "angular"
    lim = "" if limits is None else f"""
        float physics:lowerLimit = {limits[0]:.6f}
        float physics:upperLimit = {limits[1]:.6f}"""
    return f"""
    def Physics{kind}Joint "{name}" (
        prepend apiSchemas = ["PhysicsDriveAPI:{api}"]
    )
    {{
        uniform token drive:{api}:physics:type = "force"
        float drive:{api}:physics:stiffness = 0
        float drive:{api}:physics:damping = 0
        rel physics:body0 = </{root}/{b0}>
        rel physics:body1 = </{root}/{b1}>
        uniform token physics:axis = "{axis}"
        point3f physics:localPos0 = {_f3(pos0)}
        quatf physics:localRot0 = {_f4(rot0)}
        point3f physics:localPos1 = {_f3(pos1)}
        quatf physics:localRot1 = {_f4(rot1)}{lim}
    }}"""


def header(root: str, looks: set[str], friction: tuple[float, float], root_api: str = "", root_body: str = "") -> str:
    materials = "".join(uk._material_usda(k, root, LOOKS[k]) for k in sorted(looks))
    return f"""#usda 1.0
(
    defaultPrim = "{root}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}"{root_api}
{{{root_body}
    def Material "PhysicsMaterial" (
        prepend apiSchemas = ["PhysicsMaterialAPI"]
    )
    {{
        float physics:staticFriction = {friction[0]:.3f}
        float physics:dynamicFriction = {friction[1]:.3f}
        float physics:restitution = 0
    }}

    def Scope "Looks"
    {{{materials}
    }}
"""


def _rod_inertia(m: float, r: float, length: float) -> tuple[float, float, float]:
    side = m * (3 * r * r + length * length) / 12
    return (max(side, 1e-8), max(side, 1e-8), max(m * r * r / 2, 1e-9))


# -- the rose bush ----------------------------------------------------------------------------------------------------


def _pot_parts(rng) -> list[tuple[str, Mesh, str]]:
    rb, rt, h = POT_R_BOTTOM, POT_R_TOP, POT_H
    shell = lathe([(0.0, 0.0), (rb - 0.004, 0.0), (rb, 0.004), (rt - 0.006, h - 0.013), (rt, h - 0.012), (rt + 0.002, h - 0.006),
                   (rt + 0.001, h), (rt - 0.009, h), (rt - 0.010, SOIL_Z - 0.002), (0.0, SOIL_Z - 0.002)], 48)
    foot = lathe([(0.0, -0.0005), (rb - 0.002, -0.0005), (rb - 0.002, 0.0015), (0.0, 0.0015)], 32)
    soil = []
    n, rings = 40, 6
    r_out = rt - 0.0095
    verts, faces = [(0.0, 0.0, SOIL_Z + 0.004)], []
    for i in range(1, rings + 1):
        r = r_out * i / rings
        for k in range(n):
            a = 2 * math.pi * k / n
            dz = 0.0035 * (1 - (i / rings) ** 2) + rng.uniform(-0.0012, 0.0012)
            verts.append((r * math.cos(a), r * math.sin(a), SOIL_Z + dz))
    for k in range(n):
        faces.append((0, 1 + k, 1 + (k + 1) % n))
    for i in range(rings - 1):
        a0, a1 = 1 + i * n, 1 + (i + 1) * n
        for k in range(n):
            j = (k + 1) % n
            faces += [(a0 + k, a1 + k, a1 + j), (a0 + k, a1 + j, a0 + j)]
    soil.append(Mesh(np.array(verts), np.array(faces)))
    chips, grit = [], []
    for _ in range(26):
        r = r_out * math.sqrt(rng.uniform(0.05, 0.92))
        a = rng.uniform(0, 2 * math.pi)
        p = (r * math.cos(a), r * math.sin(a), SOIL_Z + 0.0015)
        if math.hypot(p[0] - CANE_IN_POT[0], p[1] - CANE_IN_POT[1]) < 0.012:
            continue
        if rng.uniform() < 0.6:
            chips.append(spun(box((rng.uniform(0.008, 0.016), rng.uniform(0.005, 0.009), 0.0025), bevel=0.0006), rng.uniform(0, 3.14)).moved(p))
        else:
            grit.append(lathe([(0.0, -0.001), (0.0025, 0.0), (0.0015, 0.002), (0.0, 0.0025)], 7).moved(p))
    parts = [("shell", shell, "terracotta"), ("foot", foot, "terracotta_dark"), ("soil", join(*soil), "soil"), ("chips", join(*chips), "bark")]
    if grit:
        parts.append(("grit", join(*grit), "grit"))
    return parts


def _green_cane_parts(rng) -> list[tuple[str, Mesh, str]]:
    canes, thorns, leaves, young = [], [], [], []
    flower: list[tuple[Mesh, str]] = []
    for k, (base, top, r) in enumerate(GREEN_CANES):
        b = np.array([base[0], base[1], SOIL_Z - 0.004])
        t = np.array(top)
        mid = (b + t) / 2 + np.array([rng.uniform(-0.004, 0.004), rng.uniform(-0.004, 0.004), 0.0])
        pts = [b + (mid - b) * s for s in np.linspace(0, 1, 5)] + [mid + (t - mid) * s for s in np.linspace(0, 1, 5)[1:]]
        canes.append(uk.tube(pts, r, 12))
        d = (t - b) / np.linalg.norm(t - b)
        for j, s in enumerate(np.linspace(0.12, 0.85, 6)):
            p = b + (t - b) * s
            a = rng.uniform(0, 2 * math.pi) + j * 2.3
            out = np.array([math.cos(a), math.sin(a), 0.0])
            out -= out.dot(d) * d
            thorns.append(thorn(p + out / np.linalg.norm(out) * r * 0.8, out, size=1.1))
        for j, s in enumerate((0.45, 0.62, 0.80, 0.94)):
            p = b + (t - b) * s
            if p[2] < 0.15:
                continue
            a = rng.uniform(0, 2 * math.pi) + j * 2.4
            leaf = compound_leaf(p, a, rng.uniform(-0.25, 0.15), rng.uniform(0.85, 1.1), rng)
            (young if s > 0.9 else leaves).append(leaf)
        if k == 1:
            flower = rose_flower(t + np.array([0.0, 0.0, 0.004]), 1.0)
        elif k == 3:
            flower.append((lathe([(0.0, 0.0), (0.006, 0.006), (0.005, 0.014), (0.0, 0.022)], 10).moved(t), "petal_dark"))
            sepal = lathe([(0.0, 0.0), (0.003, 0.003), (0.001, 0.016), (0.0, 0.019)], 6).scaled((1.0, 0.3, 1.0))
            sepal = Mesh(sepal.v @ _rx(-0.5).T, sepal.f)
            flower += [(spun(sepal, 2 * math.pi * i / 5).moved(t), "sepal") for i in range(5)]
    parts = [("canes", join(*canes), "cane"), ("thorns", join(*thorns), "thorn"), ("leaves", join(*leaves), "leaf")]
    if young:
        parts.append(("young_leaves", join(*young), "leaf_young"))
    by_look: dict[str, list[Mesh]] = {}
    for m, look in flower:
        by_look.setdefault(look, []).append(m)
    parts += [(f"flower_{look}", join(*ms), look) for look, ms in by_look.items()]
    return parts


def _dead_cane_fixed_parts() -> list[tuple[str, Mesh, str]]:
    """The dead cane's part that never moves: a short stump in the soil, with two thorns low down (below any cut)."""
    stump = placed(join(rod(-0.008, 0.006, R0 + 0.0004), cap(0.006, R0 + 0.0004, True)), CANE_IN_POT, CANE_DIR)
    up = np.array([0.0, -CANE_DIR[2], CANE_DIR[1]])  # across the cane, on its upper side
    thorns = join(thorn(on_cane(0.014) + np.array([R0 * 0.8, 0.0, 0.0]), (1.0, 0.0, 0.0), up=0.0, size=1.0),
                  thorn(on_cane(0.030) + up * R0 * 0.8, up - 0.3 * CANE_DIR, up=0.0, size=0.9))
    return [("dead_stump", stump, "cane"), ("dead_thorns", thorns, "thorn")]


def bud_mesh() -> Mesh:
    """The bud, pointing -x out of the cane's side, in a frame on the cane's axis at the bud's height."""
    m = lathe([(0.0, -0.0015), (0.0030, -0.0005), (0.0026, 0.0020), (0.0011, 0.0034), (0.0, 0.0040)], 12)
    m = Mesh(m.v @ _ry(-math.pi / 2 + 0.35).T, m.f)  # lean up a little, as buds do
    return m.moved((-(R0 - 0.0015), 0.0, 0.0))


def _lean() -> tuple[np.ndarray, tuple[float, float, float, float]]:
    rot = _rx(-LEAN)  # turns +z onto CANE_DIR
    return rot, _q_wxyz_from_matrix(rot)


def write_bush() -> str:
    root = "Bush"
    rng = np.random.default_rng(RNG_SEED)
    rot_lean, q_lean = _lean()
    q_back = (q_lean[0], -q_lean[1], -q_lean[2], -q_lean[3])
    pot_parts = _pot_parts(rng) + _green_cane_parts(rng) + _dead_cane_fixed_parts()
    pot_hull = lathe([(0.0, 0.0), (POT_R_BOTTOM, 0.0), (POT_R_TOP + 0.001, POT_H), (0.0, POT_H)], 24)
    pot_colliders = hull("pot_c", pot_hull, root)
    for k, (base, top, r) in enumerate(GREEN_CANES):
        pot_colliders += segment_box(f"cane{k}_c", (base[0], base[1], SOIL_Z), top, r, root)
    used = {p[2] for p in pot_parts}

    # The stub: two sleeves and the bud, each sliding along the cane's axis.
    sg = [("side", rod(0.0, L_SG, R0), "cane"), ("cut_face", cap(L_SG, R0, True), "pith")]
    sb = [("side", rod(0.0, L_SB, R1), "dead"), ("cut_face", cap(L_SB, R1, True), "pith_dead"),
          ("dark_band", rod(0.0, 0.004, R1 + 0.00008), "dead_dark")]
    bud = [("bud", bud_mesh(), "bud")]
    # The cutting: its top piece, two sleeves inside it, and its own bud.
    top = [("side", rod(0.0, L_T, R0), "dead"), ("old_cut", cap(L_T, R0, True), "old_cut"), ("cut_face", cap(0.0, R0, False), "pith_dead"),
           ("thorns", join(thorn((R0 * 0.8, 0.0, 0.012), (1.0, 0.0, 0.0), size=1.1), thorn((-0.002, R0 * 0.8, 0.031), (-0.3, 1.0, 0.0), size=1.0),
                           thorn((-R0 * 0.75, -0.0015, 0.047), (-1.0, -0.3, 0.0), size=0.9)), "thorn")]
    cb = [("side", rod(0.0, L_CB, R1), "dead"), ("cut_face", cap(0.0, R1, False), "pith_dead")]
    cg = [("side", rod(0.0, L_CG, R2), "cane"), ("cut_face", cap(0.0, R2, False), "pith")]
    cbud = [("bud", bud_mesh().moved(BUD_HIDDEN), "bud")]
    for parts in (sg, sb, bud, top, cb, cg, cbud):
        used |= {p[2] for p in parts}

    text = header(root, used, (0.9, 0.8))
    pot_body = pot_colliders + visuals(pot_parts, root)
    text += link("pot", 3.0, (5.5e-3, 5.5e-3, 1.0e-2), pot_body, com=(0.0, 0.0, 0.03),
                 extra_api=', "PhysicsArticulationRootAPI", "PhysxArticulationAPI"',
                 extra="\n        bool physxArticulation:enabledSelfCollisions = 1\n        int physxArticulation:solverPositionIterationCount = 32"
                       "\n        int physxArticulation:solverVelocityIterationCount = 4")

    def cane_box(length: float, r: float, z0: float = 0.0) -> str:
        return obox("c", (2 * r, 2 * r, length), (0.0, 0.0, z0 + length / 2), root)

    def apart(*names: str) -> tuple[str, str]:
        targets = ", ".join(f"</{root}/{n}>" for n in names)
        return ', "PhysicsFilteredPairsAPI"', f"\n        rel physics:filteredPairs = [{targets}]"

    cutting = ("cut_top", "cut_brown", "cut_green")
    api, rel = apart("stub_brown", *cutting)
    text += link("stub_green", 0.003, _rod_inertia(0.003, R0, L_SG), cane_box(L_SG, R0) + visuals(sg, root), extra_api=api, extra=rel,
                 at=on_cane(SG_BOTTOM_S), orient=q_lean, gravity=False, com=(0, 0, L_SG / 2))
    api, rel = apart(*cutting)
    text += link("stub_brown", 0.002, _rod_inertia(0.002, R1, L_SB), cane_box(L_SB, R1) + visuals(sb, root), extra_api=api, extra=rel,
                 at=on_cane(BROWN_S), orient=q_lean, gravity=False, com=(0, 0, L_SB / 2))
    text += link("stub_bud", 0.0002, (1e-9, 1e-9, 1e-9), visuals(bud, root), at=on_cane(BUD_S), orient=q_lean, gravity=False)
    o_c = tuple(on_cane(T_LOW_S))
    for name in ("c_px", "c_py", "c_pz", "c_rx", "c_ry"):
        text += link(name, 1.0e-4, (1e-9, 1e-9, 1e-9), "", at=o_c, gravity=False)
    text += link("cut_top", 0.004, _rod_inertia(0.004, R0, L_T), cane_box(L_T, R0) + visuals(top, root), at=o_c, orient=q_lean, com=(0, 0, L_T / 2))
    api, rel = apart("cut_green")
    text += link("cut_brown", 0.003, _rod_inertia(0.003, R1, L_CB), cane_box(L_CB, R1) + visuals(cb, root), extra_api=api, extra=rel,
                 at=on_cane(T_LOW_S + CB_HIDE), orient=q_lean, com=(0, 0, L_CB / 2))
    text += link("cut_green", 0.002, _rod_inertia(0.002, R2, L_CG), cane_box(L_CG, R2) + visuals(cg, root),
                 at=on_cane(T_LOW_S + CG_HIDE), orient=q_lean, com=(0, 0, L_CG / 2))
    text += link("cut_bud", 0.0002, (1e-9, 1e-9, 1e-9), visuals(cbud, root), at=o_c, orient=q_lean)

    d = np.subtract(BUD_SHOWN, BUD_HIDDEN)
    x_axis = d / np.linalg.norm(d)
    z_ref = np.array([0.0, 1.0, 0.0])
    z_axis = np.cross(x_axis, z_ref)
    z_axis /= np.linalg.norm(z_axis)
    bud_rot = _q_wxyz_from_matrix(np.stack([x_axis, np.cross(z_axis, x_axis), z_axis], axis=1))
    j = ""
    j += joint("Prismatic", "stub_green_slide", root, "pot", "stub_green", "Z", on_cane(SG_BOTTOM_S), q_lean, limits=(-0.09, 0.001))
    j += joint("Prismatic", "stub_brown_slide", root, "pot", "stub_brown", "Z", on_cane(BROWN_S), q_lean, limits=(-0.20, 0.001))
    j += joint("Prismatic", "stub_bud_slide", root, "pot", "stub_bud", "Z", on_cane(BUD_S), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), q_back,
               limits=(BUD_SINK - 0.005, 0.001))
    j += joint("Prismatic", "c_px_joint", root, "pot", "c_px", "X", o_c, limits=(-0.9, 0.9))
    j += joint("Prismatic", "c_py_joint", root, "c_px", "c_py", "Y", limits=(-0.9, 0.9))
    j += joint("Prismatic", "c_pz_joint", root, "c_py", "c_pz", "Z", limits=(-0.6, 0.6))
    j += joint("Revolute", "c_rx_joint", root, "c_pz", "c_rx", "X")
    j += joint("Revolute", "c_ry_joint", root, "c_rx", "c_ry", "Y")
    j += joint("Revolute", "c_rz_joint", root, "c_ry", "cut_top", "Z", (0.0, 0.0, 0.0), q_lean)
    j += joint("Prismatic", "cut_brown_slide", root, "cut_top", "cut_brown", "Z", (0.0, 0.0, CB_HIDE), limits=(-0.09, 0.001))
    j += joint("Prismatic", "cut_green_slide", root, "cut_top", "cut_green", "Z", (0.0, 0.0, CG_HIDE), limits=(-0.12, 0.001))
    j += joint("Prismatic", "cut_bud_slide", root, "cut_top", "cut_bud", "X", (0.0, 0.0, 0.0), bud_rot, (0.0, 0.0, 0.0), bud_rot,
               limits=(-0.001, BUD_TRAVEL + 0.002))
    text += j + "\n}\n"
    os.makedirs(ASSETS, exist_ok=True)
    path = ASSETS / "bush.usda"
    path.write_text(text)
    return str(path)


# -- the secateurs ----------------------------------------------------------------------------------------------------


def _closed_half(side: int) -> tuple[list[tuple[str, Mesh, str]], list[tuple[str, tuple, tuple]]]:
    """One half of the secateurs as it sits when fully closed. side=+1: the blade half (blade on +x in front of the pivot, handle
    on -x behind it); side=-1: the hook half. Returns visual parts and colliders (name, size, centre), in the tool frame."""
    s = side
    parts: list[tuple[str, Mesh, str]] = []
    hx = -s * HANDLE_X
    # the grip: a red, rounded, slightly swelling sleeve
    y0, y1 = GRIP_Y
    grip = lathe([(0.0, 0.0), (0.0062, 0.0), (0.0080, 0.006), (0.0084, 0.045), (0.0088, 0.075), (0.0080, 0.090), (0.0050, 0.094), (0.0, 0.094)], 16)
    grip = grip.scaled((HANDLE_W / 0.0168, HANDLE_T / 0.0168, 1.0))
    grip = Mesh(grip.v @ _rx(math.pi / 2).T, grip.f).moved((hx, y0, 0.0))  # along -y from y0
    parts.append(("grip", grip, "grip"))
    # the bare steel neck from the pivot to the grip
    neck = box((0.011, abs(y0) - 0.006, 0.010), center=(hx * 0.62, (y0 - 0.006) / 2 - 0.003, 0.0), bevel=0.0015, round_xy=0.002)
    parts.append(("neck", neck, "steel_dark"))
    colliders = [("grip_c", (HANDLE_W, abs(y1 - y0), HANDLE_T), (hx, (y0 + y1) / 2, 0.0)),
                 ("neck_c", (0.011, abs(y0) - 0.006, 0.010), (hx * 0.62, (y0 - 0.006) / 2 - 0.003, 0.0))]
    edge_a = s * OVERLAP / 2  # the edge's angle from +y, turned toward -x for the blade (side +1) when closed
    e = np.array([-math.sin(edge_a), math.cos(edge_a)])
    n = np.array([math.cos(edge_a), math.sin(edge_a)]) * s  # from the edge into the jaw's metal
    if side == 1:
        # the blade: a curved, sharp steel blade, 3 mm thick, above the middle
        pts = []
        for k in range(15):
            u = k / 14
            pts.append(e * (-0.010 + 0.068 * u) + n * (0.0 + 0.0012 * math.sin(math.pi * u)))
        for k in range(15):
            u = 1 - k / 14
            pts.append(e * (-0.012 + 0.066 * u) + n * (0.0045 + 0.012 * math.sin(math.pi * min(1.0, u * 1.08)) ** 0.8))
        outline = np.array(pts)
        if np.cross(outline[1] - outline[0], outline[-1] - outline[0]) < 0:
            outline = outline[::-1]
        parts.append(("blade", prism(outline, BLADE_Z - 0.0014, BLADE_Z + 0.0014, bevel=0.0004), "steel"))
        bevel_line = np.array([e * (-0.005 + 0.06 * k / 9) + n * 0.0018 for k in range(10)])
        parts.append(("edge", join(*(box((0.0012, 0.0062, 0.0004)).turned(math.degrees(edge_a), "z").moved((p[0], p[1], BLADE_Z + 0.0015))
                                     for p in bevel_line)), "steel_dark"))
    else:
        # the hook (anvil): a thicker dark jaw with a curled tip, below the middle
        pts = []
        for k in range(13):
            u = k / 12
            hook_tip = 0.004 * max(0.0, (u - 0.78) / 0.22) ** 1.5
            pts.append(e * (-0.010 + 0.052 * u) + n * (0.0 - hook_tip))
        for k in range(13):
            u = 1 - k / 12
            pts.append(e * (-0.012 + 0.050 * u) + n * (0.006 + 0.010 * math.sin(math.pi * min(1.0, u * 1.05)) ** 0.7))
        outline = np.array(pts)
        if np.cross(outline[1] - outline[0], outline[-1] - outline[0]) < 0:
            outline = outline[::-1]
        parts.append(("hook", prism(outline, -0.0058, -0.0006, bevel=0.0006), "anvil"))
        # the pivot bolt and its nut, the boss round it, and the safety catch behind it
        parts.append(("bolt", join(lathe([(0.0, -0.0075), (0.0068, -0.0075), (0.0068, -0.0058), (0.0, -0.0058)], 20),
                                   lathe([(0.0, 0.0034), (0.0062, 0.0034), (0.0062, 0.0062), (0.0, 0.0062)], 6)), "brass"))
        parts.append(("boss", lathe([(0.0, -0.0058), (0.0105, -0.0058), (0.0105, 0.0034), (0.0, 0.0034)], 24), "steel_dark"))
        parts.append(("catch", box((0.007, 0.012, 0.004), center=(0.0, -0.016, 0.0072), bevel=0.001, round_xy=0.002), "grip"))
        colliders.append(("boss_c", (0.021, 0.021, 0.013), (0.0, 0.0, -0.0012)))
    return parts, colliders


def _turned(parts, colliders, angle: float):
    c, s = math.cos(angle), math.sin(angle)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    out_parts = [(n, Mesh(m.v @ rot.T, m.f), look) for n, m, look in parts]
    out_cols = [(n, size, tuple(rot @ np.asarray(ctr, float)), _q_wxyz_z(angle)) for n, size, ctr in colliders]
    return out_parts, out_cols


def write_secateurs() -> str:
    root = "Secateurs"
    blade_parts, blade_cols = _turned(*_closed_half(+1), -HALF)  # at rest, open: the blade half turned clockwise
    hook_parts, hook_cols = _turned(*_closed_half(-1), +HALF)
    used = {p[2] for p in blade_parts + hook_parts}
    text = header(root, used, (1.0, 0.9))

    def body(parts, cols, tag):
        return "".join(obox(f"{tag}_{n}", size, ctr, root, q, patch=0.012 if n.startswith("grip") else 0.0) for n, size, ctr, q in cols) + \
            visuals([(f"{tag}_{n}", m, look) for n, m, look in parts], root)

    inertia = (2.7e-4, 1.2e-5, 2.8e-4)
    text += link("hook_half", 0.11, inertia, body(hook_parts, hook_cols, "h"), com=(0.0, -0.025, 0.0),
                 extra_api=', "PhysicsArticulationRootAPI", "PhysxArticulationAPI"',
                 extra="\n        bool physxArticulation:enabledSelfCollisions = 0\n        int physxArticulation:solverPositionIterationCount = 32"
                       "\n        int physxArticulation:solverVelocityIterationCount = 4")
    text += link("blade_half", 0.09, inertia, body(blade_parts, blade_cols, "b"), com=(0.0, -0.025, 0.0))
    text += joint("Revolute", "pivot", root, "hook_half", "blade_half", "Z", limits=(0.0, math.degrees(CLOSE)))
    text += "\n}\n"
    path = ASSETS / "secateurs.usda"
    path.write_text(text)
    return str(path)


# -- the bin ----------------------------------------------------------------------------------------------------------


def write_bin() -> str:
    root = "Bin"
    rb, rt, h, w = BIN_R_BOTTOM, BIN_R_TOP, BIN_H, BIN_WALL
    shell = lathe([(0.0, 0.0), (rb - 0.006, 0.0), (rb, 0.006), (rt, h - 0.008), (rt + 0.006, h - 0.006), (rt + 0.006, h), (rt - w, h),
                   (rt - w - 0.0005, h - 0.008), (rb - w, w + 0.004), (rb - w - 0.006, w), (0.0, w)], 48)
    ribs = join(*(box((0.004, 0.004, h - 0.03), center=((rb + rt) / 2 + 0.0005, 0.0, h / 2 - 0.004)).turned(math.degrees(a), "z")
                  for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)))
    handles = join(*(box((0.010, 0.05, 0.010), center=(0.0, 0.0, h - 0.016), bevel=0.002, round_xy=0.003).moved((s * (rt + 0.007), 0.0, 0.0))
                     for s in (-1, 1)))
    parts = [("shell", shell, "bin"), ("ribs", ribs, "bin"), ("handles", handles, "bin")]
    cols = obox("bottom_c", (2 * rb * 0.92, 2 * rb * 0.92, w), (0.0, 0.0, w / 2), root, indent="    ")
    n = 16
    for k in range(n):
        a = 2 * math.pi * (k + 0.5) / n
        r = (rb + rt) / 2 - w / 2
        tilt = math.atan2(rt - rb, h)
        size = (w + 0.002, 2 * math.pi * ((rb + rt) / 2) / n * 1.08, h)
        rot = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]]) @ _ry(tilt)
        cols += obox(f"wall{k}_c", size, (r * math.cos(a), r * math.sin(a), h / 2), root, _q_wxyz_from_matrix(rot), indent="    ")
    text = header(root, {"bin"}, (0.6, 0.5), root_api=' (\n    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]\n)',
                  root_body="\n    float physics:mass = 0.6\n    int physxRigidBody:solverPositionIterationCount = 16\n    float physxRigidBody:maxDepenetrationVelocity = 1\n")
    text += cols + f"""
    def Xform "visual"
    {{{visuals(parts, root)}
    }}
}}
"""
    path = ASSETS / "bin.usda"
    path.write_text(text)
    return str(path)


def write_all() -> None:
    write_bush()
    write_secateurs()
    write_bin()


# -- the cut, in joint positions ----------------------------------------------------------------------------------------


def cut_joint_positions(h: float) -> dict[str, float]:
    """Each slide's position once the dead cane is cut h along it from the soil (clamped to what the parts can show)."""
    h = min(max(h, H_MIN_S), T_LOW_S - 0.003)
    q = {"stub_green_slide": min(h, BROWN_S) - BROWN_S}
    q["stub_brown_slide"] = (h if h >= BROWN_S else h - 0.001) - (BROWN_S + L_SB)
    q["stub_bud_slide"] = 0.0 if h >= BUD_TOP_S else BUD_SINK
    q["cut_brown_slide"] = (max(h, BROWN_S) - T_LOW_S) - CB_HIDE
    q["cut_green_slide"] = (h - T_LOW_S) - CG_HIDE if h < BROWN_S else 0.0
    q["cut_bud_slide"] = BUD_TRAVEL if h < BUD_TOP_S else 0.0
    return q


def stub_joint_positions(h: float, bud_present: bool) -> dict[str, float]:
    """Shorten only the retained stub; the first cutting keeps its own shape."""
    h = max(h, 0.006)
    return {"stub_green_slide": min(h, BROWN_S) - BROWN_S,
            "stub_brown_slide": (h if h >= BROWN_S else h - 0.001) - (BROWN_S + L_SB),
            "stub_bud_slide": 0.0 if bud_present else BUD_SINK}
