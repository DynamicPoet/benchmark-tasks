"""The flask, the stir bar, the beaker and the stir-bar positioner/retriever, as plain-text USD.

Frames (metres, z up):
- flask, beaker: origin on the table under the middle of the base.
- bar: origin at its middle; its long axis along the bar frame's x.
- tool: origin at the tip of the tube, on its axis; the tube runs up +z from it. Two links: "tube" (the articulation's
  root) and "rod" (the magnet rod inside the tube, with its stem and ring handle above the tube's top), joined by the
  prismatic joint "slide" along z: 0 with the rod pushed fully down (magnet at the tip), up to ROD_TRAVEL with it slid up.

Glass walls collide through convex hull segments (a ring of wedges per height band), so the bar and the tube really pass
through the neck. Pure Python and NumPy.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from usdkit import Look, Mesh, _material_usda, _mesh_usda, cylinder, join, lathe, physics_material, tube

HERE = Path(__file__).resolve().parent

# -- the parts (makers' figures where published; see the README) ---------------------------------------------------------

# DURAN narrow-neck Erlenmeyer flask, 500 mL: base 105 mm across, 180 mm tall, neck 34 mm outside
FLASK_R = 0.0525
FLASK_H = 0.180
NECK_R = 0.017
NECK_BORE = 0.0145          # 29 mm bore (34 mm less a 2.5 mm wall each side; the bore is not published)
WALL = 0.0025
FLOOR = 0.004               # glass under the inside floor
HEEL_Z = 0.012              # the straight heel at the base
SHOULDER_Z = 0.135          # where the cone meets the neck
FLASK_MASS = 0.20 + 0.30    # about 200 g of glass and 300 mL of water, the water as rigid ballast
FLASK_COM = (0.0, 0.0, 0.035)

# PYREX 1000-250 beaker, 250 mL low form: about 70 mm across and 95 mm tall
BEAKER_R = 0.035
BEAKER_H = 0.095
BEAKER_WALL = 0.002
BEAKER_FLOOR = 0.003
BEAKER_MASS = 0.11

# SP Bel-Art Spinbar F37109-0010: red PTFE-coated Alnico octagon bar, 25.4 x 8 mm, with a pivot ring
BAR_L = 0.0254
BAR_R = 0.004
BAR_MASS = 0.004

# SP Bel-Art positioner/retriever H37780-0000: polyethylene tube 15.9 mm x 305 mm, a magnet rod sliding inside it
TUBE_R = 0.00795
TUBE_L = 0.305
ROD_TRAVEL = 0.030          # the rod's slide, read off the maker's photo
STEM = (0.010, 0.012)       # the rod's stem above the tube's top: across, long (from the photo)
RING_OD = 0.022             # the ring handle (from the photo)
RING_T = 0.004
RING_Z = TUBE_L + STEM[1] + RING_OD / 2    # the ring's centre above the tip
TUBE_MASS = 0.055
ROD_MASS = 0.030
ROD_COM_Z = 0.06            # the magnet sits at the rod's lower end

LOOKS = {
    "glass": Look(color=(0.90, 0.96, 0.98), roughness=0.02, opacity=0.12),
    "pe": Look(color=(0.92, 0.92, 0.88), roughness=0.5, opacity=0.92),
    "ptfe": Look(color=(0.80, 0.05, 0.04), roughness=0.4),   # red PTFE
    "grad": Look(color=(0.95, 0.95, 0.95), roughness=0.6),
}

GLASS_FRICTION = (0.5, 0.4)
PTFE_FRICTION = (0.15, 0.10)   # PTFE is slick on glass
PE_FRICTION = (0.5, 0.4)


# -- colliders --------------------------------------------------------------------------------------------------------


def _fmt(a, digits: int = 6) -> str:
    return ", ".join("(" + ", ".join(f"{x:.{digits}f}" for x in row) + ")" for row in np.asarray(a))


def hull(name: str, points, root: str, offset: float = 0.001) -> str:
    """A convex hull collider through `points` (not rendered)."""
    p = np.asarray(points, float)
    # any closed face set will do: PhysX cooks the hull from the points
    n = len(p)
    tris = [(0, i, i + 1) for i in range(1, n - 1)]
    return f"""
    def Mesh "{name}" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "PhysicsMeshCollisionAPI", "MaterialBindingAPI"]
    )
    {{
        rel material:binding:physics = </{root}/PhysicsMaterial>
        uniform token purpose = "guide"
        uniform token physics:approximation = "convexHull"
        float physxCollision:contactOffset = {offset}
        float physxCollision:restOffset = 0
        int[] faceVertexCounts = [{", ".join("3" for _ in tris)}]
        int[] faceVertexIndices = [{", ".join(str(i) for t in tris for i in t)}]
        point3f[] points = [{_fmt(p)}]
    }}"""


def capsule(name: str, radius: float, height: float, centre, axis: str, root: str, offset: float = 0.001) -> str:
    cx, cy, cz = centre
    return f"""
    def Capsule "{name}" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
    )
    {{
        rel material:binding:physics = </{root}/PhysicsMaterial>
        uniform token purpose = "guide"
        uniform token axis = "{axis}"
        double radius = {radius:.6f}
        double height = {height:.6f}
        float physxCollision:contactOffset = {offset}
        float physxCollision:restOffset = 0
        double3 xformOp:translate = ({cx:.6f}, {cy:.6f}, {cz:.6f})
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}"""


def cube(name: str, size, centre, root: str, offset: float = 0.001) -> str:
    (cx, cy, cz), (sx, sy, sz) = centre, size
    return f"""
    def Cube "{name}" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
    )
    {{
        rel material:binding:physics = </{root}/PhysicsMaterial>
        uniform token purpose = "guide"
        double size = 1
        float physxCollision:contactOffset = {offset}
        float physxCollision:restOffset = 0
        double3 xformOp:translate = ({cx:.6f}, {cy:.6f}, {cz:.6f})
        float3 xformOp:scale = ({sx:.6f}, {sy:.6f}, {sz:.6f})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }}"""


def wall_hulls(prefix: str, bands, root: str, n: int = 12) -> list[str]:
    """A wall of revolution as n wedges per band. `bands`: [(z0, r_in0, r_out0, z1, r_in1, r_out1)], radii as the
    flats' distance from the axis (so the inside is never narrower than r_in)."""
    k = 1.0 / math.cos(math.pi / n)
    out = []
    for b, (z0, ri0, ro0, z1, ri1, ro1) in enumerate(bands):
        for i in range(n):
            pts = []
            for t in (2 * math.pi * i / n, 2 * math.pi * (i + 1) / n):
                c, s = math.cos(t), math.sin(t)
                pts += [(k * ri0 * c, k * ri0 * s, z0), (k * ro0 * c, k * ro0 * s, z0),
                        (k * ri1 * c, k * ri1 * s, z1), (k * ro1 * c, k * ro1 * s, z1)]
            out.append(hull(f"{prefix}{b}_{i}", pts, root))
    return out


def bar_hull(root: str) -> str:
    """The octagon bar as one convex hull along x: eight flats 8 mm across (one lying on the floor), chamfered ends, so it
    lies still on a flat like the real bar instead of rolling."""
    k = 1.0 / math.cos(math.pi / 8)
    pts = []
    for x, r in ((BAR_L / 2 - 0.0015, BAR_R), (-(BAR_L / 2 - 0.0015), BAR_R), (BAR_L / 2, 0.7 * BAR_R), (-BAR_L / 2, 0.7 * BAR_R)):
        for i in range(8):
            t = math.pi / 8 + i * math.pi / 4
            pts.append((x, k * r * math.cos(t), k * r * math.sin(t)))
    return hull("c_bar", pts, root, offset=0.0008)


def disc_hull(name: str, r: float, z0: float, z1: float, root: str, n: int = 12) -> str:
    k = 1.0 / math.cos(math.pi / n)
    pts = [(k * r * math.cos(2 * math.pi * i / n), k * r * math.sin(2 * math.pi * i / n), z) for z in (z0, z1)
           for i in range(n)]
    return hull(name, pts, root)


# -- shapes -----------------------------------------------------------------------------------------------------------


def _cone_r(z: float) -> float:
    """The flask's outside radius at height z."""
    if z <= HEEL_Z:
        return FLASK_R
    if z <= SHOULDER_Z:
        return FLASK_R + (NECK_R - FLASK_R) * (z - HEEL_Z) / (SHOULDER_Z - HEEL_Z)
    return NECK_R


def flask_bands() -> list:
    zs = [FLOOR, HEEL_Z, 0.053, 0.094, SHOULDER_Z, FLASK_H]
    bands = []
    for z0, z1 in zip(zs[:-1], zs[1:]):
        ro0, ro1 = _cone_r(z0), _cone_r(z1)
        if z0 >= SHOULDER_Z:
            ri0 = ri1 = NECK_BORE
        else:
            ri0, ri1 = ro0 - WALL, max(ro1 - WALL, NECK_BORE)
        bands.append((z0, ri0, ro0, z1, ri1, ro1))
    return bands


def flask_parts() -> list:
    out = [(FLASK_R, 0.0), (FLASK_R, HEEL_Z), (NECK_R, SHOULDER_Z), (NECK_R, FLASK_H - 0.004), (NECK_R + 0.0015, FLASK_H - 0.002),
           (NECK_R, FLASK_H)]
    inner = [(NECK_BORE, FLASK_H), (NECK_BORE, SHOULDER_Z), (FLASK_R - WALL, HEEL_Z), (FLASK_R - WALL, FLOOR)]
    profile = [(0.0, 0.0)] + out + inner + [(0.0, FLOOR)]
    glass = lathe(profile, 48)
    marks = [cylinder(0.0007, 0.012, (FLASK_R * 0.8 * math.cos(0.3), FLASK_R * 0.8 * math.sin(0.3), z), segments=6)
             for z in ()]
    parts = [("glass", glass, "glass")]
    return parts + [(f"m{i}", m, "grad") for i, m in enumerate(marks)]


def beaker_parts() -> list:
    r, h = BEAKER_R, BEAKER_H
    profile = [(0.0, 0.0), (r, 0.0), (r, h - 0.002), (r + 0.0015, h), (r - BEAKER_WALL, h), (r - BEAKER_WALL, BEAKER_FLOOR),
               (0.0, BEAKER_FLOOR)]
    glass = lathe(profile, 48)
    # white graduations on the side facing out
    marks = [cylinder(0.0006, 0.010, (r * math.cos(0.0) + 0.0003, 0.0, z), axis="y", segments=6) for z in (0.03, 0.045, 0.06)]
    return [("glass", glass, "glass")] + [(f"g{i}", m, "grad") for i, m in enumerate(marks)]


def bar_parts() -> list:
    body = lathe([(0.0, -BAR_L / 2), (BAR_R * 0.7, -BAR_L / 2), (BAR_R, -BAR_L / 2 + 0.0015), (BAR_R, BAR_L / 2 - 0.0015),
                  (BAR_R * 0.7, BAR_L / 2), (0.0, BAR_L / 2)], 8).turned(90.0, "y")
    ring = cylinder(BAR_R + 0.0006, 0.0015, (0, 0, 0), axis="x", segments=16)
    return [("bar", join(body, ring), "ptfe")]


def tube_parts() -> list:
    body = cylinder(TUBE_R, TUBE_L - 0.002, (0, 0, TUBE_L / 2), segments=24, bevel=0.0015)
    return [("tube", body, "pe")]


def rod_parts() -> list:
    stem = cylinder(STEM[0] / 2, STEM[1], (0, 0, TUBE_L + STEM[1] / 2), segments=16)
    a = np.linspace(0, 2 * np.pi, 33)
    rr = (RING_OD - RING_T) / 2
    ring = tube(np.c_[rr * np.sin(a), np.zeros_like(a), RING_Z + rr * np.cos(a)][:-1], RING_T / 2, sides=10)
    return [("stem", stem, "pe"), ("ring", ring, "pe")]


# -- USD files --------------------------------------------------------------------------------------------------------


def _visual(root: str, parts: list) -> tuple[str, str]:
    used = sorted({p[2] for p in parts})
    looks = "".join(_material_usda(m, root, LOOKS[m]) for m in used)
    meshes = "".join(_mesh_usda(p[0], p[1], p[2], root, LOOKS[p[2]], 35.0) for p in parts)
    return looks, meshes


def _rigid(path: Path, root: str, parts: list, colliders: list[str], mass: float, friction, com=None,
           solver_iters: int = 32, max_depen: float = 0.3) -> None:
    looks, meshes = _visual(root, parts)
    com_line = "" if com is None else f"\n    point3f physics:centerOfMass = ({com[0]}, {com[1]}, {com[2]})"
    text = f"""#usda 1.0
(
    defaultPrim = "{root}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
)
{{
    float physics:mass = {mass:.6f}{com_line}
    int physxRigidBody:solverPositionIterationCount = {solver_iters}
    int physxRigidBody:solverVelocityIterationCount = 4
    float physxRigidBody:maxDepenetrationVelocity = {max_depen}
{physics_material("PhysicsMaterial", *friction)}
{"".join(colliders)}

    def Scope "Looks"
    {{{looks}
    }}

    def Xform "visual"
    {{{meshes}
    }}
}}
"""
    _put(path, text)


def _put(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def _link(name: str, root: str, parts: list, colliders: list[str], mass: float, com_z: float, articulation_root: bool) -> str:
    path = f"{root}/{name}"
    looks, meshes = _visual(path, parts)
    schemas = '"PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"'
    extra = ""
    if articulation_root:
        schemas += ', "PhysicsArticulationRootAPI", "PhysxArticulationAPI"'
        extra = """
    bool physxArticulation:enabledSelfCollisions = 0
    int physxArticulation:solverPositionIterationCount = 32
    int physxArticulation:solverVelocityIterationCount = 4"""
    return f"""
def Xform "{name}" (
    prepend apiSchemas = [{schemas}]
)
{{
    float physics:mass = {mass:.6f}
    point3f physics:centerOfMass = (0, 0, {com_z:.4f})
    float physxRigidBody:maxDepenetrationVelocity = 0.3{extra}
{physics_material("PhysicsMaterial", *PE_FRICTION)}
{"".join(colliders)}

    def Scope "Looks"
    {{{looks}
    }}

    def Xform "visual"
    {{{meshes}
    }}
}}
"""


def write_tool(path: Path) -> None:
    root = "tool"
    tube_cols = [capsule("c_tube", TUBE_R, TUBE_L - 2 * TUBE_R, (0, 0, TUBE_L / 2), "Z", f"{root}/tube")]
    rod_cols = [cube("c_stem", (STEM[0], STEM[0], STEM[1]), (0, 0, TUBE_L + STEM[1] / 2), f"{root}/rod"),
                cube("c_ring", (RING_OD, RING_T, RING_OD), (0, 0, RING_Z), f"{root}/rod")]
    body = (_link("tube", root, tube_parts(), tube_cols, TUBE_MASS, TUBE_L / 2, True)
            + _link("rod", root, rod_parts(), rod_cols, ROD_MASS, ROD_COM_Z, False))
    body = "\n".join("    " + line if line else line for line in body.splitlines())
    text = f"""#usda 1.0
(
    defaultPrim = "{root}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}"
{{
{body}

    def PhysicsPrismaticJoint "slide"
    {{
        uniform token physics:axis = "Z"
        rel physics:body0 = </{root}/tube>
        rel physics:body1 = </{root}/rod>
        point3f physics:localPos0 = (0, 0, 0)
        point3f physics:localPos1 = (0, 0, 0)
        quatf physics:localRot0 = (1, 0, 0, 0)
        quatf physics:localRot1 = (1, 0, 0, 0)
        float physics:lowerLimit = 0
        float physics:upperLimit = {ROD_TRAVEL}
    }}
}}
"""
    _put(path, text)


def write_all() -> None:
    a = HERE / "assets"
    fcols = [disc_hull("c_floor", FLASK_R, 0.0, FLOOR, "flask")] + wall_hulls("c_w", flask_bands(), "flask")
    _rigid(a / "flask.usda", "flask", flask_parts(), fcols, FLASK_MASS, GLASS_FRICTION, com=FLASK_COM)
    bz = [BEAKER_FLOOR, 0.034, 0.065, BEAKER_H]
    bbands = [(z0, BEAKER_R - BEAKER_WALL, BEAKER_R, z1, BEAKER_R - BEAKER_WALL, BEAKER_R) for z0, z1 in zip(bz[:-1], bz[1:])]
    bcols = [disc_hull("c_floor", BEAKER_R, 0.0, BEAKER_FLOOR, "beaker")] + wall_hulls("c_w", bbands, "beaker")
    _rigid(a / "beaker.usda", "beaker", beaker_parts(), bcols, BEAKER_MASS, GLASS_FRICTION, com=(0.0, 0.0, 0.03))
    _rigid(a / "bar.usda", "bar", bar_parts(), [bar_hull("bar")], BAR_MASS, PTFE_FRICTION, solver_iters=32, max_depen=0.2)
    write_tool(a / "tool.usda")


if __name__ == "__main__":
    write_all()
