"""The dustpan, the hand brush and the cereal rings: meshes, materials and colliders, written as plain-text USD.

Every handled object is a mesh with a real object's shape and size. Each collides through separate convex shapes kept
out of the render (purpose "guide"), within 2 mm of its visual where an arm or a ring touches it.

Frames (all metres):
- dustpan: origin at the middle of the lip's front edge, on the table; +y points into the pan, x runs along the lip.
- brush: origin at the middle of the bristles' tips, on the table; x runs along the brush, its handle towards -x.
- ring: origin at its centre.

Pure Python and NumPy (PIL for the textures).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from usdkit import Look, Mesh, _material_usda, _mesh_usda, box, cylinder, join, physics_material, prism

HERE = Path(__file__).resolve().parent

# -- dustpan: a plastic hand dustpan, 25 cm wide, 21.5 cm deep, its handle 12 cm long ----------------------------------

PAN_W = 0.25            # outer width along the lip
PAN_D = 0.215           # from the lip's front edge to the back wall's outer face
PAN_WALL = 0.0025       # side and back wall thickness
PAN_FLOOR = 0.002       # floor thickness
LIP_D = 0.03            # the rubber lip: a strip sloping from the front edge up to the floor
LIP_EDGE = 0.0004       # the lip's thickness at its front edge
SIDE_Y0 = 0.012         # the side walls start this far behind the front edge
SIDE_H0, SIDE_H1 = 0.006, 0.075  # side wall height at its front end and at the back
BACK_H = 0.075
HANDLE_L = 0.12
HANDLE_W = 0.028
HANDLE_Z = (0.055, 0.070)
PAN_MASS = 0.20         # kg: a typical polypropylene hand dustpan
GRIP_Y = PAN_D + 0.08   # where the left arm pinches the handle
INNER_HALF_W = PAN_W / 2 - PAN_WALL
INNER_BACK = PAN_D - PAN_WALL

# -- hand brush: 28 cm long, bristle block 18 x 4 cm ---------------------------------------------------------------------

BRISTLE = (0.18, 0.04, 0.025)       # length, width, height of the bristle block
BRISTLE_TIP_W = 0.024               # width across the tips: the outer rows' tips lean in, so the long edges are bevelled
BRISTLE_BEVEL_H = 0.007             # ... up to this height
BACK = (0.184, 0.046, 0.015)        # the wooden back the bristles are set in
HANDLE_X = (-0.192, -0.092)         # the handle runs from the back's end towards -x
HANDLE_WB = 0.026
HANDLE_ZB = (0.026, 0.042)
BRUSH_GRIP_X = -0.125               # where the right arm pinches the handle, 3.3 cm from the block
BRUSH_MASS = 0.11                   # kg

# -- cereal ring: toasted oat ring, 12 mm across, 5 mm thick, a 4 mm hole ----------------------------------------------

RING_R = 0.004          # radius of the ring's centre line
RING_TUBE = 0.002       # radius of its cross-section in the plane
RING_HALF_T = 0.0025    # half its thickness
RING_OUTER = RING_R + RING_TUBE

LOOKS = {
    "pan": Look(color=(0.16, 0.34, 0.62), roughness=0.45, texture="plastic.png", tile=0.05),
    "rubber": Look(color=(0.05, 0.05, 0.055), roughness=0.8),
    "wood": Look(color=(1.0, 1.0, 1.0), roughness=0.55, texture="wood.png", tile=0.12),
    "bristle": Look(color=(1.0, 1.0, 1.0), roughness=0.85, texture="bristle.png", tile=0.03),
    "oat": Look(color=(0.70, 0.50, 0.35), roughness=0.9, texture="oat.png", tile=0.01),  # toasted, darker than the table
}


# -- textures ------------------------------------------------------------------------------------------------------------


def _noise(n: int, seed: int, f0: float, width: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    spec = np.fft.fft2(rng.normal(0.0, 1.0, (n, n)))
    fx, fy = np.fft.fftfreq(n)[None, :], np.fft.fftfreq(n)[:, None]
    out = np.real(np.fft.ifft2(spec * np.exp(-((np.hypot(fx, fy) - f0) / width) ** 2)))
    return out / (3.0 * out.std())


def make_textures() -> None:
    """Textures made from noise: beech grain, black polypropylene fibres, plastic and toasted oat."""
    from PIL import Image

    folder = HERE / "textures"
    folder.mkdir(exist_ok=True)

    def save(name: str, img: np.ndarray) -> None:
        Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(folder / name)

    n = 512
    if not (folder / "wood.png").exists():
        rng = np.random.default_rng(11)
        y = np.linspace(0, 1, n)[:, None]
        x = np.linspace(0, 1, n)[None, :]
        warp = 0.02 * _noise(n, 12, 0.01, 0.01)
        grain = np.sin(2 * np.pi * (34 * (y + warp) + 0.3 * np.sin(2 * np.pi * x)))
        fine = rng.normal(0.0, 1.0, (n, 1)) * 0.5
        tone = 0.5 + 0.18 * grain + 0.10 * fine + 0.05 * _noise(n, 13, 0.05, 0.05)
        base = np.array([0.80, 0.62, 0.42])
        save("wood.png", base[None, None, :] * (0.82 + 0.30 * tone[:, :, None]))
    if not (folder / "bristle.png").exists():
        rng = np.random.default_rng(21)
        streak = rng.normal(0.0, 1.0, (1, n)) + 0.3 * rng.normal(0.0, 1.0, (n, n))
        tone = 0.13 + 0.05 * np.clip(streak, -2, 2)
        save("bristle.png", np.repeat(tone[:, :, None], 3, axis=2) * np.array([1.0, 1.0, 1.05]))
    if not (folder / "plastic.png").exists():
        tone = 1.0 + 0.04 * _noise(n, 31, 0.2, 0.1)
        save("plastic.png", np.repeat(np.clip(tone, 0, 1)[:, :, None] * 0.98, 3, axis=2))
    if not (folder / "oat.png").exists():
        rng = np.random.default_rng(41)
        spots = _noise(n, 42, 0.08, 0.06) + 0.5 * rng.normal(0.0, 1.0, (n, n))
        base = np.array([0.88, 0.66, 0.36])
        img = base[None, None, :] * (0.92 + 0.12 * np.clip(spots, -2, 2)[:, :, None])
        save("oat.png", img)


# -- meshes --------------------------------------------------------------------------------------------------------------


def _x_prism(profile_yz, x0: float, x1: float) -> Mesh:
    """A solid from a convex, counter-clockwise (y, z) profile, extruded along x from x0 to x1."""
    m = prism(profile_yz, x0, x1)
    return Mesh(m.v[:, [2, 0, 1]], m.f)  # (p, q, r) -> (x=r, y=p, z=q): a rotation, so faces stay outward


def torus(r: float, tube: float, half_t: float, seg: int = 24, sides: int = 12) -> Mesh:
    th = np.linspace(0.0, 2 * np.pi, seg, endpoint=False)
    ph = np.linspace(0.0, 2 * np.pi, sides, endpoint=False)
    v = []
    for t in th:
        for p in ph:
            rr = r + tube * np.cos(p)
            v.append((rr * np.cos(t), rr * np.sin(t), half_t * np.sin(p)))
    f = []
    for i in range(seg):
        for j in range(sides):
            a = i * sides + j
            b = ((i + 1) % seg) * sides + j
            c = ((i + 1) % seg) * sides + (j + 1) % sides
            d = i * sides + (j + 1) % sides
            f += [(a, b, c), (a, c, d)]
    return Mesh(np.array(v), np.array(f))


def frustum(c0, r0: float, c1, r1: float, segments: int = 10) -> Mesh:
    """A closed, tapered tube from a circle at c0 (radius r0) to one at c1 (radius r1), both level."""
    th = np.linspace(0.0, 2 * np.pi, segments, endpoint=False)
    c0, c1 = np.asarray(c0, float), np.asarray(c1, float)
    ring0 = c0 + r0 * np.c_[np.cos(th), np.sin(th), np.zeros_like(th)]
    ring1 = c1 + r1 * np.c_[np.cos(th), np.sin(th), np.zeros_like(th)]
    v = np.vstack([ring0, ring1, c0[None], c1[None]])
    n = segments
    f = []
    for i in range(n):
        j = (i + 1) % n
        f += [(i, j, n + j), (i, n + j, n + i), (2 * n, j, i), (2 * n + 1, n + i, n + j)]
    return Mesh(v, np.array(f))


def pan_parts() -> list:
    lip = _x_prism([(0.0, 0.0), (LIP_D, 0.0), (LIP_D, PAN_FLOOR), (0.0, LIP_EDGE)], -PAN_W / 2, PAN_W / 2)
    floor = _x_prism([(LIP_D, 0.0), (INNER_BACK, 0.0), (INNER_BACK, PAN_FLOOR), (LIP_D, PAN_FLOOR)],
                     -PAN_W / 2, PAN_W / 2)
    side = [(SIDE_Y0, 0.0), (PAN_D, 0.0), (PAN_D, SIDE_H1), (SIDE_Y0, SIDE_H0)]
    walls = join(_x_prism(side, -PAN_W / 2, -PAN_W / 2 + PAN_WALL), _x_prism(side, PAN_W / 2 - PAN_WALL, PAN_W / 2))
    back = box((PAN_W, PAN_WALL, BACK_H), (0.0, PAN_D - PAN_WALL / 2, BACK_H / 2))
    rim = box((PAN_W + 0.004, 0.012, 0.004), (0.0, PAN_D - 0.005, BACK_H - 0.002), bevel=0.001)
    handle = box((HANDLE_W, HANDLE_L + 0.01, HANDLE_Z[1] - HANDLE_Z[0]),
                 (0.0, PAN_D + HANDLE_L / 2 - 0.005, sum(HANDLE_Z) / 2), bevel=0.003, round_xy=0.008)
    gusset = _x_prism([(PAN_D - 0.002, 0.03), (PAN_D + 0.05, HANDLE_Z[0] + 0.001), (PAN_D + 0.05, HANDLE_Z[0] + 0.004),
                       (PAN_D - 0.002, HANDLE_Z[0] + 0.004)], -0.004, 0.004)
    hole = cylinder(0.0055, 0.0005, (0.0, PAN_D + HANDLE_L - 0.022, HANDLE_Z[1] + 0.0002), segments=20)
    return [("lip", lip, "rubber"), ("floor", floor, "pan"), ("walls", walls, "pan"), ("back", back, "pan"),
            ("rim", rim, "pan"), ("handle", handle, "pan"), ("gusset", gusset, "pan"), ("hole", hole, "rubber")]


def brush_parts() -> list:
    tufts = []
    top = BRISTLE[2]
    tip_y = BRISTLE_TIP_W / 2 - 0.003  # the outer rows' tips lean in to the bevel
    for y in (-0.0125, 0.0, 0.0125):
        for k in range(15):
            x = -0.084 + k * 0.012 + (0.006 if y == 0.0 else 0.0) * (k < 14)
            if y == 0.0:
                tufts.append(cylinder(0.0058, top - 0.0005, (x, y, top / 2 + 0.0005), segments=10, bevel=0.0012))
            else:
                tufts.append(frustum((x, np.sign(y) * tip_y, 0.0005), 0.003, (x, y, top), 0.0058))
    back = box(BACK, (0.0, 0.0, BRISTLE[2] + BACK[2] / 2), bevel=0.003, round_xy=0.01)
    hl = HANDLE_X[1] - HANDLE_X[0] + 0.012
    handle = box((hl, HANDLE_WB, HANDLE_ZB[1] - HANDLE_ZB[0]), ((HANDLE_X[0] + HANDLE_X[1] + 0.012) / 2, 0.0,
                 sum(HANDLE_ZB) / 2), bevel=0.004, round_xy=0.011)
    hole = cylinder(0.0045, 0.0005, (HANDLE_X[0] + 0.014, 0.0, HANDLE_ZB[1] + 0.0002), segments=16)
    return [("bristles", join(*tufts), "bristle"), ("back", back, "wood"), ("handle", handle, "wood"),
            ("hole", hole, "rubber")]


# -- colliders -----------------------------------------------------------------------------------------------------------


def _hull_faces(points: np.ndarray) -> np.ndarray:
    from scipy.spatial import ConvexHull

    hull = ConvexHull(points)
    faces = []
    centre = points.mean(axis=0)
    for tri in hull.simplices:
        a, b, c = points[tri]
        if np.dot(np.cross(b - a, c - a), a - centre) < 0:
            tri = tri[[0, 2, 1]]
        faces.append(tri)
    return np.array(faces)


def convex(name: str, points, root: str, material: str, contact_offset: float = 0.002) -> str:
    """A convex-hull collider through `points` (not rendered: purpose "guide")."""
    p = np.asarray(points, float)
    f = _hull_faces(p)
    pts = ", ".join(f"({x:.6f}, {y:.6f}, {z:.6f})" for x, y, z in p)
    return f"""
    def Mesh "{name}" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "PhysicsMeshCollisionAPI", "MaterialBindingAPI"]
    )
    {{
        uniform token physics:approximation = "convexHull"
        rel material:binding:physics = </{root}/{material}>
        uniform token purpose = "guide"
        float physxCollision:contactOffset = {contact_offset}
        float physxCollision:restOffset = 0
        int[] faceVertexCounts = [{", ".join("3" for _ in f)}]
        int[] faceVertexIndices = [{", ".join(str(int(i)) for i in f.reshape(-1))}]
        point3f[] points = [{pts}]
    }}"""


def box_points(lo, hi) -> np.ndarray:
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    return np.array([(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float)


LIP_SEGMENTS = 10   # PhysX cooks no GPU hull for a shape longer than about 100 times its thickness, so the lip is split


def cube(name: str, lo, hi, root: str, material: str, contact_offset: float = 0.002) -> str:
    """A box collider from corner lo to corner hi (a PhysX box, not rendered: purpose "guide")."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    (cx, cy, cz), (sx, sy, sz) = (lo + hi) / 2, hi - lo
    return f"""
    def Cube "{name}" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
    )
    {{
        rel material:binding:physics = </{root}/{material}>
        uniform token purpose = "guide"
        double size = 1
        float physxCollision:contactOffset = {contact_offset}
        float physxCollision:restOffset = 0
        double3 xformOp:translate = ({cx:.6f}, {cy:.6f}, {cz:.6f})
        float3 xformOp:scale = ({sx:.6f}, {sy:.6f}, {sz:.6f})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }}"""


def pan_colliders() -> list[tuple[str, np.ndarray, str, str]]:
    """The pan's collision pieces in its frame, (name, corner points, "box" or "hull", material); the scorer uses them too."""
    hw = PAN_W / 2
    out = []
    edges = np.linspace(-hw, hw, LIP_SEGMENTS + 1)
    for i in range(LIP_SEGMENTS):
        wedge = np.array([(x, y, z) for x in edges[i:i + 2] for (y, z) in ((0.0, 0.0), (0.0, LIP_EDGE), (LIP_D, 0.0),
                                                                          (LIP_D, PAN_FLOOR))])
        out.append((f"c_lip{i:02d}", wedge, "hull", "RubberMaterial"))
    out.append(("c_floor", box_points((-hw, LIP_D - 0.001, 0.0), (hw, INNER_BACK + 0.001, PAN_FLOOR)), "box",
                "PhysicsMaterial"))
    mid = (SIDE_Y0 + PAN_D) / 2
    h_mid = SIDE_H0 + (SIDE_H1 - SIDE_H0) * (mid - SIDE_Y0) / (PAN_D - SIDE_Y0)
    for half, (y0, h0, y1, h1) in enumerate(((SIDE_Y0, SIDE_H0, mid, h_mid), (mid, h_mid, PAN_D, SIDE_H1))):
        side = [(y0, 0.0), (y1, 0.0), (y1, h1), (y0, h0)]
        for name, x0 in (("c_left", -hw), ("c_right", hw - PAN_WALL)):
            pts = np.array([(x, y, z) for x in (x0, x0 + PAN_WALL) for (y, z) in side])
            out.append((f"{name}{half}", pts, "hull", "PhysicsMaterial"))
    out.append(("c_back", box_points((-hw, INNER_BACK, 0.0), (hw, PAN_D, BACK_H)), "box", "PhysicsMaterial"))
    out.append(("c_handle", box_points((-HANDLE_W / 2, PAN_D, HANDLE_Z[0]), (HANDLE_W / 2, PAN_D + HANDLE_L, HANDLE_Z[1])),
                "box", "PhysicsMaterial"))
    return out


def brush_colliders() -> list[tuple[str, np.ndarray, str]]:
    bl, bw, bh = BRISTLE
    kl, kw, kh = BACK
    return [
        ("c_bristles", np.array([(x, y, z) for x in (-bl / 2, bl / 2) for (y, z) in (
            (-BRISTLE_TIP_W / 2, 0.0), (BRISTLE_TIP_W / 2, 0.0), (-bw / 2, BRISTLE_BEVEL_H), (bw / 2, BRISTLE_BEVEL_H),
            (-bw / 2, bh), (bw / 2, bh))]), "BristleMaterial"),
        ("c_back", box_points((-kl / 2, -kw / 2, bh), (kl / 2, kw / 2, bh + kh)), "PhysicsMaterial"),
        ("c_handle", box_points((HANDLE_X[0], -HANDLE_WB / 2, HANDLE_ZB[0]), (HANDLE_X[1], HANDLE_WB / 2, HANDLE_ZB[1])),
         "PhysicsMaterial"),
    ]


def ring_hull() -> np.ndarray:
    """The ring's convex hull: the outer half of its cross-section, turned about its axis (60 points)."""
    pts = []
    for t in np.linspace(0.0, 2 * np.pi, 12, endpoint=False):
        for p in np.radians([-90.0, -45.0, 0.0, 45.0, 90.0]):
            rr = RING_R + RING_TUBE * np.cos(p)
            pts.append((rr * np.cos(t), rr * np.sin(t), RING_HALF_T * np.sin(p)))
    return np.array(pts)


# -- USD ----------------------------------------------------------------------------------------------------------------


def _write(path: Path, root: str, parts: list, colliders: list[str], mass: float, materials: str, *,
           solver_iters: int, max_depen: float, sleep: float | None = None) -> None:
    used = sorted({p[2] for p in parts})
    looks = "".join(_material_usda(m, root, LOOKS[m]) for m in used)
    meshes = "".join(_mesh_usda(p[0], p[1], p[2], root, LOOKS[p[2]], 35.0) for p in parts)
    sleep_line = f"\n    float physxRigidBody:sleepThreshold = {sleep}" if sleep is not None else ""
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
    float physics:mass = {mass:.6f}
    int physxRigidBody:solverPositionIterationCount = {solver_iters}
    int physxRigidBody:solverVelocityIterationCount = 4
    float physxRigidBody:maxDepenetrationVelocity = {max_depen}{sleep_line}
{materials}
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


# Friction (static, dynamic). The station's table uses PhysX's default material (0.5, 0.5); PhysX averages the two.
PAN_FRICTION = (0.35, 0.30)       # polypropylene
RUBBER_FRICTION = (0.8, 0.7)      # the lip's rubber strip
WOOD_FRICTION = (0.5, 0.45)       # lacquered beech
BRISTLE_FRICTION = (0.35, 0.30)   # polypropylene bristle tips sliding
OAT_FRICTION = (0.45, 0.40)       # dry toasted cereal
RING_MASS = 0.0003                # kg


def write_all() -> None:
    make_textures()
    root = "dustpan"
    mats = physics_material("PhysicsMaterial", *PAN_FRICTION) + physics_material("RubberMaterial", *RUBBER_FRICTION)
    cols = [cube(n, p.min(axis=0), p.max(axis=0), root, m) if kind == "box" else convex(n, p, root, m)
            for n, p, kind, m in pan_colliders()]
    _write(HERE / "assets" / "dustpan.usda", root, pan_parts(), cols, PAN_MASS, mats, solver_iters=32, max_depen=0.5)

    root = "brush"
    mats = physics_material("PhysicsMaterial", *WOOD_FRICTION) + physics_material("BristleMaterial", *BRISTLE_FRICTION)
    cols = [convex(n, p, root, m) for n, p, m in brush_colliders()]
    _write(HERE / "assets" / "brush.usda", root, brush_parts(), cols, BRUSH_MASS, mats, solver_iters=32, max_depen=0.5)

    root = "ring"
    mats = physics_material("PhysicsMaterial", *OAT_FRICTION)
    cols = [convex("c_ring", ring_hull(), root, "PhysicsMaterial", contact_offset=0.0015)]
    parts = [("ring", torus(RING_R, RING_TUBE, RING_HALF_T), "oat")]
    _write(HERE / "assets" / "ring.usda", root, parts, cols, RING_MASS, mats, solver_iters=16, max_depen=0.3)
