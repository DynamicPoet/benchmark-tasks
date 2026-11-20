"""The scene's parts as plain-text USD: six 10 mL vials (five Lock Flush, one Injection), two hang-and-stack bins and a
plastic tray. Run on import by task.py (`write_all()`); the files only change when their text changes.

Sizes: the bins are Akro-Mils AkroBin 30210, 5-3/8 x 4-1/8 x 3 in outside (137 x 105 x 76 mm); the tray is a Cambro
Camtray 1014, 27 x 35 cm (rim height taken as 15 mm). The vials' maker publishes no size: they are drawn as a standard
10 mL serum vial, 24.5 mm across and 45 mm to the shoulder, with a crimp and flip cap, 59.5 mm tall in all.
Each vial collides through two convex hulls (body; neck and cap), the bins and the tray through boxes and hulls.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import usdkit as uk
from usdkit import Look, Mesh, _material_usda, _mesh_usda, physics_material

HERE = Path(__file__).resolve().parent

# -- the vial ---------------------------------------------------------------------------------------------------------
VIAL_R = 0.01225
BODY_H = 0.045
NECK_R = 0.0075
CAP_R = 0.0100
VIAL_H = 0.0595
VIAL_MASS = 0.025          # glass about 13 g, about 10 mL water, cap: an estimate
LABEL_R = 0.01235
LABEL_Z0, LABEL_Z1 = 0.006, 0.041
LABEL_ARC = math.radians(300.0)   # the label wraps 300 degrees; its printed panel faces the vial's +x
GLASS_FRICTION = (0.6, 0.5)

# -- the bins (AkroBin 30210) -------------------------------------------------------------------------------------------
BIN_L, BIN_W, BIN_H = 0.137, 0.105, 0.076   # along x (front to back), along y, tall
BIN_FRONT_H = 0.038                         # the open scoop front's lip
BIN_WALL = 0.004                            # collider thickness (the real wall is thinner)
BIN_FLOOR = 0.003

# -- the tray (Camtray 1014) --------------------------------------------------------------------------------------------
TRAY_L, TRAY_W = 0.35, 0.27
TRAY_RIM = 0.015
TRAY_FLOOR = 0.003

LOOKS = {
    "glass": Look(color=(0.80, 0.84, 0.86), roughness=0.08, metallic=0.0),
    "crimp": Look(color=(0.72, 0.72, 0.74), roughness=0.3, metallic=0.9),
    "cap": Look(color=(0.95, 0.42, 0.08), roughness=0.45),
    "bin_blue": Look(color=(0.05, 0.22, 0.62), roughness=0.55),
    "bin_red": Look(color=(0.66, 0.06, 0.05), roughness=0.55),
    "tray": Look(color=(0.62, 0.63, 0.62), roughness=0.5),
}


def _fmt(a, digits: int = 6) -> str:
    return ", ".join("(" + ", ".join(f"{x:.{digits}f}" for x in row) + ")" for row in np.asarray(a))


def hull(name: str, points, root: str, offset: float = 0.001) -> str:
    p = np.asarray(points, float)
    tris = [(0, i, i + 1) for i in range(1, len(p) - 1)]
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


def cube(name: str, size, centre, root: str, offset: float = 0.002) -> str:
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


def disc_hull(name: str, r: float, z0: float, z1: float, root: str, n: int = 16, offset: float = 0.001) -> str:
    k = 1.0 / math.cos(math.pi / n)
    pts = [(k * r * math.cos(2 * math.pi * i / n), k * r * math.sin(2 * math.pi * i / n), z) for z in (z0, z1)
           for i in range(n)]
    return hull(name, pts, root, offset)


# -- meshes with their own texture coordinates --------------------------------------------------------------------------


def textured_mesh(name: str, v, f, uv, root: str, texture: str, indent: str = "        ") -> tuple[str, str]:
    """A mesh whose corners carry the given (u, v) in 0..1 of one texture (not usdkit's box projection)."""
    v, f, uv = np.asarray(v, float), np.asarray(f, int), np.asarray(uv, float)
    mat = f"tex_{name}"
    look = Look(color=(1.0, 1.0, 1.0), roughness=0.5, texture=texture, tile=1.0)
    normals = uk._corner_normals(v, f, 35.0).reshape(-1, 3)
    corner_uv = uv[f].reshape(-1, 2)
    mesh = f"""
{indent}def Mesh "{name}" (
{indent}    prepend apiSchemas = ["MaterialBindingAPI"]
{indent})
{indent}{{
{indent}    rel material:binding = </{root}/Looks/{mat}>
{indent}    uniform bool doubleSided = 0
{indent}    int[] faceVertexCounts = [{", ".join("3" for _ in range(len(f)))}]
{indent}    int[] faceVertexIndices = [{", ".join(str(int(i)) for i in f.reshape(-1))}]
{indent}    point3f[] points = [{_fmt(v, 5)}]
{indent}    normal3f[] normals = [{_fmt(normals, 4)}] (
{indent}        interpolation = "faceVarying"
{indent}    )
{indent}    texCoord2f[] primvars:st = [{_fmt(corner_uv, 4)}] (
{indent}        interpolation = "faceVarying"
{indent}    )
{indent}    uniform token subdivisionScheme = "none"
{indent}}}"""
    return _material_usda(mat, root, look), mesh


def label_mesh(n: int = 60):
    """The wrap label: a cylindrical shell of radius LABEL_R over LABEL_ARC, centred on +x, u along the arc, v up."""
    v, uv = [], []
    for i in range(n + 1):
        s = i / n
        a = -LABEL_ARC / 2 + s * LABEL_ARC
        for z, t in ((LABEL_Z0, 0.0), (LABEL_Z1, 1.0)):
            v.append((LABEL_R * math.cos(a), LABEL_R * math.sin(a), z))
            uv.append((s, t))
    f = []
    for i in range(n):
        a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
        f += [(a, c, d), (a, d, b)]
    return v, f, uv


def card_mesh(w: float, h: float):
    """A flat card in the y-z plane facing -x, (u, v) 0..1 reading left to right as seen from -x."""
    v = [(0.0, w / 2, -h / 2), (0.0, -w / 2, -h / 2), (0.0, -w / 2, h / 2), (0.0, w / 2, h / 2)]
    uv = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    f = [(0, 1, 2), (0, 2, 3)]
    return v, f, uv


# -- parts ---------------------------------------------------------------------------------------------------------------


def vial_parts() -> list:
    body = uk.lathe([(0.0, 0.0), (VIAL_R - 0.0015, 0.0), (VIAL_R, 0.0015), (VIAL_R, BODY_H - 0.002),
                     (VIAL_R - 0.003, BODY_H + 0.0015), (NECK_R, BODY_H + 0.003), (NECK_R, 0.050)], segments=32)
    crimp = uk.cylinder(CAP_R, 0.0065, center=(0.0, 0.0, 0.050 + 0.00325), segments=32, bevel=0.0005)
    cap = uk.cylinder(CAP_R, 0.003, center=(0.0, 0.0, 0.0565 + 0.0015), segments=32, bevel=0.0008)
    return [("glass", body, "glass"), ("crimp", crimp, "crimp"), ("cap", cap, "cap")]


def bin_parts() -> list:
    """The bin as boxes: floor, back wall, a low front lip and two side walls that step down toward the front."""
    L, W, H, Hf, t = BIN_L, BIN_W, BIN_H, BIN_FRONT_H, 0.003
    m = [
        uk.box((L, W, t), center=(0.0, 0.0, t / 2)),
        uk.box((t, W, H), center=(L / 2 - t / 2, 0.0, H / 2)),
        uk.box((t, W, Hf), center=(-L / 2 + t / 2, 0.0, Hf / 2)),
    ]
    for s in (-1, 1):
        y = s * (W / 2 - t / 2)
        m.append(uk.box((L * 0.55, t, H), center=(L / 2 - L * 0.275, y, H / 2)))
        m.append(uk.box((L * 0.45, t, (H + Hf) / 2), center=(-L / 2 + L * 0.225, y, (H + Hf) / 4)))
    return [("bin", uk.join(*m), "BIN")]


def bin_colliders(root: str) -> list[str]:
    L, W, H, Hf, t = BIN_L, BIN_W, BIN_H, BIN_FRONT_H, BIN_WALL
    out = [cube("c_floor", (L, W, BIN_FLOOR), (0.0, 0.0, BIN_FLOOR / 2), root),
           cube("c_back", (t, W, H), (L / 2 - t / 2, 0.0, H / 2), root),
           cube("c_front", (t, W, Hf), (-L / 2 + t / 2, 0.0, Hf / 2), root)]
    for s, n in ((-1, "r"), (1, "l")):
        y = s * (W / 2 - t / 2)
        out.append(cube(f"c_side_{n}0", (L * 0.55, t, H), (L / 2 - L * 0.275, y, H / 2), root))
        out.append(cube(f"c_side_{n}1", (L * 0.45, t, (H + Hf) / 2), (-L / 2 + L * 0.225, y, (H + Hf) / 4), root))
    return out


def tray_parts() -> list:
    L, W, t, r = TRAY_L, TRAY_W, 0.005, TRAY_RIM
    m = [uk.box((L, W, TRAY_FLOOR), center=(0.0, 0.0, TRAY_FLOOR / 2), round_xy=0.02)]
    m += [uk.box((t, W, r), center=(s * (L / 2 - t / 2), 0.0, r / 2)) for s in (-1, 1)]
    m += [uk.box((L, t, r), center=(0.0, s * (W / 2 - t / 2), r / 2)) for s in (-1, 1)]
    return [("tray", uk.join(*m), "tray")]


def tray_colliders(root: str) -> list[str]:
    L, W, t, r = TRAY_L, TRAY_W, 0.008, TRAY_RIM
    out = [cube("c_floor", (L, W, TRAY_FLOOR), (0.0, 0.0, TRAY_FLOOR / 2), root)]
    out += [cube(f"c_end{i}", (t, W, r), (s * (L / 2 - t / 2), 0.0, r / 2), root) for i, s in enumerate((-1, 1))]
    out += [cube(f"c_side{i}", (L, t, r), (0.0, s * (W / 2 - t / 2), r / 2), root) for i, s in enumerate((-1, 1))]
    return out


# -- writing --------------------------------------------------------------------------------------------------------------


def _write(path: Path, root: str, parts: list, colliders: list[str], textured: list[tuple[str, str]], mass: float | None,
           friction, solver_iters: int = 32, max_depen: float = 0.5) -> None:
    used = sorted({p[2] for p in parts})
    looks = "".join(_material_usda(m, root, LOOKS[m]) for m in used)
    meshes = "".join(_mesh_usda(p[0], p[1], p[2], root, LOOKS[p[2]], 35.0) for p in parts)
    for mat, mesh in textured:
        looks += mat
        meshes += mesh
    looks = looks.replace((HERE / "textures").as_posix(), "../textures")   # relative to the file in assets/
    # every part is a rigid body; a fixed one is made kinematic by the task (UsdBody with mass=None)
    mass = 1.0 if mass is None else mass
    head = """ (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
)"""
    body = f"""
    float physics:mass = {mass:.6f}
    int physxRigidBody:solverPositionIterationCount = {solver_iters}
    int physxRigidBody:solverVelocityIterationCount = 4
    float physxRigidBody:maxDepenetrationVelocity = {max_depen}
"""
    text = f"""#usda 1.0
(
    defaultPrim = "{root}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}"{head}
{{{body}{physics_material("PhysicsMaterial", *friction)}
{"".join(colliders)}

    def Scope "Looks"
    {{{looks}
    }}

    def Xform "visual"
    {{{meshes}
    }}
}}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def write_vial(path: Path, root: str, kind: str) -> None:
    v, f, uv = label_mesh()
    tex = [textured_mesh("label", v, f, uv, root, f"label_{kind}.png")]
    cols = [disc_hull("c_body", VIAL_R, 0.0, BODY_H, root, offset=0.0015),
            disc_hull("c_cap", CAP_R, BODY_H, VIAL_H, root, offset=0.0015)]
    _write(path, root, vial_parts(), cols, tex, VIAL_MASS, GLASS_FRICTION)


def write_bin(path: Path, root: str, color: str, card_tex: str) -> None:
    LOOKS["BIN"] = LOOKS[color]
    v, f, uv = card_mesh(0.090, 0.024)
    v = np.asarray(v) + np.array([-BIN_L / 2 - 0.0006, 0.0, BIN_FRONT_H / 2])
    tex = [textured_mesh("card", v, f, uv, root, card_tex)]
    _write(path, root, bin_parts(), bin_colliders(root), tex, None, (0.5, 0.4))


def write_all() -> None:
    a = HERE / "assets"
    write_vial(a / "vial_flush.usda", "vial", "flush")
    write_vial(a / "vial_injection.usda", "vial", "injection")
    write_bin(a / "bin_flush.usda", "bin", "bin_blue", "card_flush.png")
    write_bin(a / "bin_returns.usda", "bin", "bin_red", "card_returns.png")
    _write(a / "tray.usda", "tray", tray_parts(), tray_colliders("tray"), [], None, (0.5, 0.4))


if __name__ == "__main__":
    write_all()
