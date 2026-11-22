"""usdkit: small visual meshes, photo-textured materials and simple colliders, written as plain-text USD (no pxr needed).

Everything an arm handles in this task is a mesh with a real object's shape, and collides through separate plain boxes kept out
of the render (purpose "guide"). Meshes carry per-corner normals, so a bevel catches light and a flat face stays flat, and
per-corner texture coordinates in metres, so a texture keeps its real scale on any part.

Pure Python and NumPy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


# -- meshes --------------------------------------------------------------------------------


@dataclass
class Mesh:
    v: np.ndarray  # [n, 3]
    f: np.ndarray  # [m, 3], counter-clockwise from outside

    def moved(self, t) -> "Mesh":
        return Mesh(self.v + np.asarray(t, float), self.f)

    def turned(self, deg: float, axis: str = "z") -> "Mesh":
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        r = {"x": [[1, 0, 0], [0, c, -s], [0, s, c]], "y": [[c, 0, s], [0, 1, 0], [-s, 0, c]], "z": [[c, -s, 0], [s, c, 0], [0, 0, 1]]}[axis]
        return Mesh(self.v @ np.array(r).T, self.f)

    def scaled(self, s) -> "Mesh":
        return Mesh(self.v * np.asarray(s, float), self.f)


def join(*meshes: Mesh) -> Mesh:
    vs, fs, n = [], [], 0
    for m in meshes:
        vs.append(m.v)
        fs.append(m.f + n)
        n += len(m.v)
    return Mesh(np.vstack(vs), np.vstack(fs))


def _rounded_outline(sx: float, sy: float, r: float, seg: int = 4) -> np.ndarray:
    """A rectangle's outline, counter-clockwise, its corners rounded with radius r."""
    if r <= 0:
        return np.array([[sx, -sy], [sx, sy], [-sx, sy], [-sx, -sy]], float) * 0.5
    pts = []
    hx, hy = sx / 2 - r, sy / 2 - r
    for cx, cy, a0 in ((hx, -hy, -90.0), (hx, hy, 0.0), (-hx, hy, 90.0), (-hx, -hy, 180.0)):
        for k in range(seg + 1):
            a = np.radians(a0 + 90.0 * k / seg)
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return np.array(pts)


def prism(outline, z0: float, z1: float, bevel: float = 0.0) -> Mesh:
    """A closed solid from a counter-clockwise, star-shaped outline in xy, from z0 to z1, its top and bottom edges bevelled."""
    outline = np.asarray(outline, float)
    c = outline.mean(axis=0)
    d = outline - c
    length = np.linalg.norm(d, axis=1, keepdims=True)
    inset = outline - d / np.maximum(length, 1e-9) * bevel
    rings = [(inset, z0), (outline, z0 + bevel), (outline, z1 - bevel), (inset, z1)] if bevel > 0 else [(outline, z0), (outline, z1)]
    n = len(outline)
    v = [np.c_[r, np.full(n, z)] for r, z in rings]
    v.append(np.array([[c[0], c[1], z0], [c[0], c[1], z1]]))
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


def box(size, center=(0.0, 0.0, 0.0), bevel: float = 0.0, round_xy: float = 0.0) -> Mesh:
    """A box of `size` at `center`; `bevel` chamfers the top and bottom edges, `round_xy` rounds the vertical ones."""
    sx, sy, sz = size
    m = prism(_rounded_outline(sx, sy, round_xy), -sz / 2, sz / 2, bevel)
    return m.moved(center)


def lathe(profile, segments: int = 32) -> Mesh:
    """Revolve a (radius, z) profile about z; radius 0 closes the surface. Give the profile from bottom to top, outside first."""
    th = np.linspace(0.0, 2 * np.pi, segments, endpoint=False)
    rings, verts = [], []
    for r, z in profile:
        if r == 0.0:
            rings.append([len(verts)])
            verts.append((0.0, 0.0, z))
        else:
            rings.append(list(range(len(verts), len(verts) + segments)))
            verts += [(r * np.cos(t), r * np.sin(t), z) for t in th]
    faces = []
    for a, b in zip(rings[:-1], rings[1:]):
        if len(a) == 1 and len(b) > 1:
            faces += [(a[0], b[(i + 1) % segments], b[i]) for i in range(segments)]
        elif len(b) == 1 and len(a) > 1:
            faces += [(a[i], a[(i + 1) % segments], b[0]) for i in range(segments)]
        elif len(a) > 1:
            for i in range(segments):
                j = (i + 1) % segments
                faces += [(a[i], a[j], b[j]), (a[i], b[j], b[i])]
    return Mesh(np.array(verts, float), np.array(faces))


def cylinder(radius: float, height: float, center=(0.0, 0.0, 0.0), axis: str = "z", segments: int = 24, bevel: float = 0.0) -> Mesh:
    h = height / 2
    if bevel > 0:
        prof = [(0.0, -h), (radius - bevel, -h), (radius, -h + bevel), (radius, h - bevel), (radius - bevel, h), (0.0, h)]
    else:
        prof = [(0.0, -h), (radius, -h), (radius, h), (0.0, h)]
    m = lathe(prof, segments)
    if axis == "x":
        m = m.turned(90.0, "y")
    elif axis == "y":
        m = m.turned(-90.0, "x")
    return m.moved(center)


def tube(points, radius: float, sides: int = 10) -> Mesh:
    """A closed tube along a polyline."""
    p = np.asarray(points, float)
    d = np.gradient(p, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    u = np.cross(d[0], [0.0, 0.0, 1.0] if abs(d[0][2]) < 0.9 else [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    th = np.linspace(0.0, 2 * np.pi, sides, endpoint=False)
    rings = []
    for k in range(len(p)):
        u = u - u.dot(d[k]) * d[k]
        u /= np.linalg.norm(u)
        w = np.cross(d[k], u)
        rings.append(p[k] + radius * (np.outer(np.cos(th), u) + np.outer(np.sin(th), w)))
    v = np.vstack(rings + [p[0][None], p[-1][None]])
    f = []
    for k in range(len(p) - 1):
        for i in range(sides):
            j = (i + 1) % sides
            a, b, c, e = k * sides + i, k * sides + j, (k + 1) * sides + j, (k + 1) * sides + i
            f += [(a, b, c), (a, c, e)]
    first, last = len(p) * sides, len(p) * sides + 1
    for i in range(sides):
        j = (i + 1) % sides
        f += [(first, j, i), (last, (len(p) - 1) * sides + i, (len(p) - 1) * sides + j)]
    return Mesh(v, np.array(f))


def _corner_normals(v: np.ndarray, f: np.ndarray, crease_deg: float) -> np.ndarray:
    """One normal per triangle corner: the mean over the faces round its vertex that lie within `crease_deg` of its own face."""
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    # Vertices that coincide count as one, so a seam or a join does not break the shading.
    _, ids = np.unique(np.round(v, 6), axis=0, return_inverse=True)
    ids = ids.reshape(-1)
    around: dict[int, list[int]] = {}
    for i, tri in enumerate(f):
        for c in tri:
            around.setdefault(int(ids[c]), []).append(i)
    cos_crease = np.cos(np.radians(crease_deg))
    out = np.zeros((len(f), 3, 3))
    for i, tri in enumerate(f):
        for k, c in enumerate(tri):
            faces = around[int(ids[c])]
            near = fn[faces] @ fn[i] > cos_crease
            n = fn[faces][near].sum(axis=0)
            out[i, k] = n / max(np.linalg.norm(n), 1e-12)
    return out


def _corner_uvs(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Box projection in metres. A side face maps z to v, so a texture's rows run level round the part."""
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    dom = np.argmax(np.abs(fn), axis=1)
    p = v[f]
    uv = np.zeros((len(f), 3, 2))
    uv[dom == 2] = p[dom == 2][:, :, [0, 1]]
    uv[dom == 0] = p[dom == 0][:, :, [1, 2]]
    uv[dom == 1] = p[dom == 1][:, :, [0, 2]]
    return uv


# -- materials ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Look:
    """A UsdPreviewSurface. `texture`, `rough_map` and `normal_map` are files under textures/, repeated every `tile` metres."""

    color: tuple[float, float, float] = (0.5, 0.5, 0.5)  # linear; multiplies `texture`
    metallic: float = 0.0
    roughness: float = 0.6
    texture: str | None = None
    rough_map: str | None = None
    normal_map: str | None = None
    tile: float = 0.1
    emissive: tuple[float, float, float] | None = None
    opacity: float = 1.0


def _material_usda(name: str, root: str, look: Look) -> str:
    base = f"/{root}/Looks/{name}"

    def tex(shader: str, file: str, space: str, out_type: str, scale=None, bias=None) -> str:
        extra = ""
        if scale is not None:
            extra += f"\n                float4 inputs:scale = ({scale[0]}, {scale[1]}, {scale[2]}, 1)"
        if bias is not None:
            extra += f"\n                float4 inputs:bias = ({bias[0]}, {bias[1]}, {bias[2]}, 0)"
        return f"""
            def Shader "{shader}"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @{(HERE / 'textures' / file).as_posix()}@
                float2 inputs:st.connect = <{base}/uv.outputs:result>
                token inputs:wrapS = "repeat"
                token inputs:wrapT = "repeat"
                token inputs:sourceColorSpace = "{space}"{extra}
                {out_type} outputs:{'rgb' if out_type == 'float3' else 'r'}
            }}"""

    lines = [f'uniform token info:id = "UsdPreviewSurface"', f"float inputs:metallic = {look.metallic}"]
    shaders = ""
    c = look.color
    if look.texture:
        lines.append(f"color3f inputs:diffuseColor.connect = <{base}/color_tex.outputs:rgb>")
        shaders += tex("color_tex", look.texture, "sRGB", "float3", scale=c)
    else:
        lines.append(f"color3f inputs:diffuseColor = ({c[0]}, {c[1]}, {c[2]})")
    if look.rough_map:
        lines.append(f"float inputs:roughness.connect = <{base}/rough_tex.outputs:r>")
        shaders += tex("rough_tex", look.rough_map, "raw", "float")
    else:
        lines.append(f"float inputs:roughness = {look.roughness}")
    if look.normal_map:
        lines.append(f"normal3f inputs:normal.connect = <{base}/normal_tex.outputs:rgb>")
        shaders += tex("normal_tex", look.normal_map, "raw", "float3", scale=(2, 2, 2), bias=(-1, -1, -1))
    if look.emissive:
        e = look.emissive
        lines.append(f"color3f inputs:emissiveColor = ({e[0]}, {e[1]}, {e[2]})")
    if look.opacity < 1.0:
        lines.append(f"float inputs:opacity = {look.opacity}")
    body = "\n                ".join(lines)
    if shaders:
        shaders = f"""
            def Shader "uv"
            {{
                uniform token info:id = "UsdPrimvarReader_float2"
                token inputs:varname = "st"
                float2 outputs:result
            }}""" + shaders
    return f"""
        def Material "{name}"
        {{
            token outputs:surface.connect = <{base}/surface.outputs:surface>
            def Shader "surface"
            {{
                {body}
                token outputs:surface
            }}{shaders}
        }}"""


# -- prims ----------------------------------------------------------------------------------


def _fmt(a: np.ndarray, digits: int = 5) -> str:
    return ", ".join("(" + ", ".join(f"{x:.{digits}f}" for x in row) + ")" for row in a)


def _mesh_usda(name: str, mesh: Mesh, mat: str, root: str, look: Look, crease: float, indent: str = "        ", hidden: bool = False) -> str:
    v, f = mesh.v, mesh.f
    normals = _corner_normals(v, f, crease).reshape(-1, 3)
    uv = (_corner_uvs(v, f) / look.tile).reshape(-1, 2)
    vis = f'\n{indent}    token visibility = "invisible"' if hidden else ""
    return f"""
{indent}def Mesh "{name}" (
{indent}    prepend apiSchemas = ["MaterialBindingAPI"]
{indent})
{indent}{{{vis}
{indent}    rel material:binding = </{root}/Looks/{mat}>
{indent}    uniform bool doubleSided = 0
{indent}    int[] faceVertexCounts = [{", ".join("3" for _ in range(len(f)))}]
{indent}    int[] faceVertexIndices = [{", ".join(str(int(i)) for i in f.reshape(-1))}]
{indent}    point3f[] points = [{_fmt(v)}]
{indent}    normal3f[] normals = [{_fmt(normals, 4)}] (
{indent}        interpolation = "faceVarying"
{indent}    )
{indent}    texCoord2f[] primvars:st = [{_fmt(uv, 4)}] (
{indent}        interpolation = "faceVarying"
{indent}    )
{indent}    uniform token subdivisionScheme = "none"
{indent}}}"""


def collider(name: str, size, center, root: str, material: str = "PhysicsMaterial") -> str:
    """A box collider (not rendered: purpose "guide")."""
    sx, sy, sz = size
    cx, cy, cz = center
    return f"""
    def Cube "{name}" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
    )
    {{
        rel material:binding:physics = </{root}/{material}>
        uniform token purpose = "guide"
        double size = 1
        float physxCollision:contactOffset = 0.004
        float physxCollision:restOffset = 0
        double3 xformOp:translate = ({cx:.6f}, {cy:.6f}, {cz:.6f})
        float3 xformOp:scale = ({sx:.6f}, {sy:.6f}, {sz:.6f})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }}"""


def physics_material(name: str, static: float, dynamic: float) -> str:
    return f"""
    def Material "{name}" (
        prepend apiSchemas = ["PhysicsMaterialAPI"]
    )
    {{
        float physics:staticFriction = {static:.4f}
        float physics:dynamicFriction = {dynamic:.4f}
        float physics:restitution = 0
    }}"""


def write_usda(path: str, root: str, parts, looks: dict[str, Look], colliders: list[str], *, mass: float | None = None,
               friction: tuple[float, float] = (0.6, 0.5), extra_materials: str = "", crease: float = 35.0,
               solver_iters: int = 32) -> str:
    """One prim: a rigid body when `mass` is given, otherwise fixed. `parts` is a list of (name, Mesh, look) or
    (name, Mesh, look, hidden)."""
    used = sorted({p[2] for p in parts})
    materials = "".join(_material_usda(m, root, looks[m]) for m in used)
    meshes = "".join(_mesh_usda(p[0], p[1], p[2], root, looks[p[2]], crease, hidden=len(p) > 3 and p[3]) for p in parts)
    if mass is None:
        head, body = "", ""
    else:
        head = """ (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
)"""
        body = f"""
    float physics:mass = {mass:.4f}
    int physxRigidBody:solverPositionIterationCount = {solver_iters}
    int physxRigidBody:solverVelocityIterationCount = 4
    float physxRigidBody:maxDepenetrationVelocity = 1
"""
    text = f"""#usda 1.0
(
    defaultPrim = "{root}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}"{head}
{{{body}{physics_material("PhysicsMaterial", friction[0], friction[1])}{extra_materials}
{"".join(colliders)}

    def Scope "Looks"
    {{{materials}
    }}

    def Xform "visual"
    {{{meshes}
    }}
}}
"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as out:
        out.write(text)
    return path
