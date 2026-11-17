"""The bread basket, the rolls, the broken wine glass and the waste bin: meshes, materials and colliders as plain-text USD.

Every handled object is a mesh with a real object's shape and size. Each collides through separate convex shapes kept out
of the render (purpose "guide"), within about 2 mm of its visual where an arm or another part touches it.

Frames (metres, z up):
- basket: origin on the table under the middle of its floor; its long axis along y, its two loop handles at the ends.
- roll: origin at its centre; its long axis along x.
- glass pieces: origin at each piece's centre of mass, roughly.
- bin: origin on the table under the middle of its floor.

Pure Python and NumPy (PIL for the textures).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from usdkit import Look, Mesh, _material_usda, _mesh_usda, box, join, lathe, physics_material, tube

HERE = Path(__file__).resolve().parent

# -- bread basket: a shallow oval woven willow bread tray, 30 x 22 cm at the rim, 4.5 cm tall, sides flared 50 degrees, a strap handle at each end

BASKET_RIM = (0.22, 0.30)    # outer size at the rim, along x and y
BASKET_BASE = (0.115, 0.195)  # outer size at the floor: the sides flare out about 50 degrees from vertical
BASKET_H = 0.045             # floor to rim top
BASKET_WALL = 0.007          # woven wall thickness
BASKET_FLOOR = 0.007
RIM_R = 0.006                # the rolled rim's radius
HANDLE_OUT = 0.04            # the handle bar's centre line stands this far out from the rim's end
HANDLE_LEN = 0.07            # the bar's length along x
HANDLE_T = 0.008             # the handle is a flat woven strap: this thick (along y) ...
HANDLE_W = 0.022             # ... and this tall
HANDLE_R = HANDLE_T / 2      # half its thickness
HANDLE_Z = BASKET_H - HANDLE_W / 2  # the strap's centre height: its top level with the rim's top
BASKET_MASS = 0.18

# -- bread rolls: soft white dinner rolls, golden crust ------------------------------------------------------------------

ROLL_SIZE = (0.070, 0.055, 0.040)  # length, width, height
ROLL_MASS = 0.03                    # artificial display rolls (a real 40-60 g roll is denser)

# -- the broken wine glass: clear soda-lime glass, 2 mm wall ---------------------------------------------------------------

FOOT_R = 0.035          # foot radius
FOOT_T = 0.003
STEM_R = 0.0035
STEM_L = 0.085
STUB_H = 0.02           # the jagged stub of the bowl left on the stem
BOWL_R = 0.04           # the bowl's radius where the pieces came from
GLASS_T = 0.002
STEM_PIECE_MASS = 0.07
SHARD_MASS = 0.012      # each large curved piece, about 5 x 4 cm
CHIP_MASS = 0.002       # each small piece, about 15 x 10 mm
SHARD_ARC = (0.050, 0.040)  # the large pieces' extent along the bowl's circumference and its height
CHIP = (0.015, 0.010)

# -- waste bin: a low lined plastic bin, fixed ----------------------------------------------------------------------------

BIN_OUT = (0.20, 0.36)   # outer size along x and y
BIN_H = 0.10
BIN_WALL = 0.005
BIN_FLOOR = 0.005
BIN_IN = (BIN_OUT[0] - 2 * BIN_WALL, BIN_OUT[1] - 2 * BIN_WALL)

LOOKS = {
    "wicker": Look(color=(1.0, 1.0, 1.0), roughness=0.75, texture="wicker.png", tile=0.08),
    "crust": Look(color=(1.0, 1.0, 1.0), roughness=0.7, texture="crust.png", tile=0.06),
    "glass": Look(color=(0.92, 0.96, 0.97), roughness=0.05, opacity=0.35),
    "binbody": Look(color=(0.55, 0.57, 0.6), roughness=0.5),
    "liner": Look(color=(0.03, 0.03, 0.035), roughness=0.35),
}


# -- textures ------------------------------------------------------------------------------------------------------------


def _noise(n: int, seed: int, f0: float, width: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    spec = np.fft.fft2(rng.normal(0.0, 1.0, (n, n)))
    fx, fy = np.fft.fftfreq(n)[None, :], np.fft.fftfreq(n)[:, None]
    out = np.real(np.fft.ifft2(spec * np.exp(-((np.hypot(fx, fy) - f0) / width) ** 2)))
    return out / (3.0 * out.std())


def make_textures() -> None:
    """Textures made from noise: woven willow and a golden bread crust."""
    from PIL import Image

    folder = HERE / "textures"
    folder.mkdir(exist_ok=True)

    def save(name: str, img: np.ndarray) -> None:
        Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(folder / name)

    n = 512
    if not (folder / "wicker.png").exists():
        y = np.linspace(0, 1, n, endpoint=False)[:, None]
        x = np.linspace(0, 1, n, endpoint=False)[None, :]
        rows = 16
        band = (y * rows) % 1.0
        over = ((np.floor(y * rows) + np.floor(x * 8)) % 2 == 0)
        weave = np.where(over, np.sin(np.pi * band), 0.55 + 0.3 * np.sin(np.pi * ((x * 8) % 1.0)))
        tone = 0.55 + 0.35 * weave + 0.06 * _noise(n, 3, 0.08, 0.06)
        base = np.array([0.72, 0.55, 0.33])
        save("wicker.png", base[None, None, :] * tone[:, :, None])
    if not (folder / "crust.png").exists():
        rng = np.random.default_rng(7)
        spots = _noise(n, 8, 0.03, 0.03) + 0.4 * _noise(n, 9, 0.15, 0.1) + 0.15 * rng.normal(0, 1, (n, n))
        base = np.array([0.80, 0.55, 0.25])
        save("crust.png", base[None, None, :] * (0.9 + 0.15 * np.clip(spots, -2, 2)[:, :, None]))


# -- shapes --------------------------------------------------------------------------------------------------------------


def _oval(sx: float, sy: float, n: int = 40) -> np.ndarray:
    """A superellipse outline (a rounded oval), counter-clockwise, sizes along x and y."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    e = 2.6
    return np.c_[sx / 2 * np.sign(c) * np.abs(c) ** (2 / e), sy / 2 * np.sign(s) * np.abs(s) ** (2 / e)]


def _loft(outlines: list[np.ndarray], zs: list[float], cap_bottom: bool = False, cap_top: bool = False) -> Mesh:
    """A surface through closed outlines of the same length at heights zs."""
    n = len(outlines[0])
    v = [np.c_[o, np.full(n, z)] for o, z in zip(outlines, zs)]
    f = []
    for k in range(len(outlines) - 1):
        for i in range(n):
            j = (i + 1) % n
            a, b, c, d = k * n + i, k * n + j, (k + 1) * n + j, (k + 1) * n + i
            f += [(a, b, c), (a, c, d)]
    v = np.vstack(v)
    if cap_bottom:
        v = np.vstack([v, [[0.0, 0.0, zs[0]]]])
        ci = len(v) - 1
        f += [(ci, (i + 1) % n, i) for i in range(n)]
    if cap_top:
        v = np.vstack([v, [[0.0, 0.0, zs[-1]]]])
        ci = len(v) - 1
        last = (len(outlines) - 1) * n
        f += [(ci, last + i, last + (i + 1) % n) for i in range(n)]
    return Mesh(v, np.array(f))


def _basket_outline(z: float) -> np.ndarray:
    """The basket's outer outline at height z: it flares from the base to the rim."""
    s = np.clip(z / BASKET_H, 0.0, 1.0)
    return _oval(BASKET_BASE[0] + s * (BASKET_RIM[0] - BASKET_BASE[0]), BASKET_BASE[1] + s * (BASKET_RIM[1] - BASKET_BASE[1]))


def basket_parts() -> list:
    zs = [0.0, BASKET_H - RIM_R]
    outer = [_basket_outline(z) for z in zs]
    inner = [o * (1 - 2 * BASKET_WALL / np.array(BASKET_RIM)) for o in outer]
    wall = two_sided(join(_loft(outer, zs, cap_bottom=True), _loft(inner[::-1], [zs[1], BASKET_FLOOR], cap_top=True)))
    rim_line = np.vstack([_basket_outline(BASKET_H - RIM_R) * (1 - BASKET_WALL / np.array(BASKET_RIM))] * 1)
    rim_pts = np.c_[rim_line, np.full(len(rim_line), BASKET_H - RIM_R)]
    rim = tube(np.vstack([rim_pts, rim_pts[:1]]), RIM_R, sides=8)
    lip = two_sided(_loft([inner[1], outer[1]], [zs[1], zs[1]]))
    parts = [("wall", wall, "wicker"), ("rim", rim, "wicker"), ("lip", lip, "wicker")]
    for side in (-1.0, 1.0):
        y_end = side * BASKET_RIM[1] / 2
        y_bar = y_end + side * HANDLE_OUT
        y_in = y_end - side * 0.004
        bar = box((HANDLE_LEN + HANDLE_T, HANDLE_T, HANDLE_W), (0.0, y_bar, HANDLE_Z), bevel=0.0015)
        arms = [box((HANDLE_T, abs(y_bar - y_in), HANDLE_W), (sx * HANDLE_LEN / 2, (y_bar + y_in) / 2, HANDLE_Z), bevel=0.0015)
                for sx in (-1.0, 1.0)]
        parts.append((f"handle_{'r' if side < 0 else 'l'}", join(bar, *arms), "wicker"))
    return parts


def roll_parts() -> list:
    lx, ly, lz = ROLL_SIZE
    prof = []
    for t in np.linspace(0, np.pi, 13):
        r = np.sin(t)
        z = -np.cos(t)
        prof.append((0.0 if r < 1e-6 else r, z))
    m = lathe(prof, 24)
    v = m.v.copy()
    v[:, 2] = np.where(v[:, 2] < 0, v[:, 2] * 0.55, v[:, 2])  # a flatter base
    v = v * np.array([lx / 2, ly / 2, lz / 2 / 1.0])
    v[:, 2] -= lz * 0.1
    return [("roll", Mesh(v, m.f), "crust")]


def _stem_components() -> list[Mesh]:
    foot = lathe([(0.0, 0.0), (FOOT_R, 0.0), (FOOT_R - 0.002, FOOT_T), (STEM_R * 2.2, FOOT_T + 0.004), (0.0, FOOT_T + 0.004)], 32)
    stem = lathe([(0.0, FOOT_T), (STEM_R, FOOT_T + 0.003), (STEM_R, FOOT_T + STEM_L), (0.0, FOOT_T + STEM_L)], 16)
    z0 = FOOT_T + STEM_L
    t = np.linspace(0, 2 * np.pi, 33)[:-1]
    jag = STUB_H * (0.55 + 0.45 * np.abs(np.sin(3 * t + 0.4)) * (0.6 + 0.4 * np.cos(5 * t)))
    rings = []
    for s in np.linspace(0, 1, 5):
        r = STEM_R + (BOWL_R - STEM_R) * np.sin(s * np.pi / 2) * 0.55
        rings.append((np.c_[r * np.cos(t), r * np.sin(t)], z0 + s * jag))
    n = len(t)
    v = np.vstack([np.c_[o, z] for o, z in rings])
    f = []
    for k in range(len(rings) - 1):
        for i in range(n):
            j = (i + 1) % n
            a, b, c, d = k * n + i, k * n + j, (k + 1) * n + j, (k + 1) * n + i
            f += [(a, b, c), (a, c, d), (a, c, b), (a, d, c)]
    stub = Mesh(v, np.array(f))
    return [m.turned(90.0, "x") for m in (foot, stem, stub)]  # lying on its side, the stem along -y from the foot


def stem_parts() -> list:
    return [("stem", join(*_stem_components()), "glass")]


def _shard_mesh(arc: float, height: float, seed: int) -> Mesh:
    """A curved piece of a 2 mm bowl wall, roughly arc x height, its edges ragged."""
    rng = np.random.default_rng(seed)
    nu, nv = 9, 7
    us = np.linspace(-arc / 2, arc / 2, nu)
    vs = np.linspace(-height / 2, height / 2, nv)
    pts_out, pts_in = [], []
    for v_ in vs:
        for u in us:
            a = u / BOWL_R
            r = BOWL_R
            pts_out.append((r * np.sin(a), v_, r * (np.cos(a) - 1.0)))
            pts_in.append(((r - GLASS_T) * np.sin(a), v_, (r - GLASS_T) * np.cos(a) - r))
    keep = np.ones((nv, nu), bool)
    for i in range(nv):
        for j in range(nu):
            edge = min(i, nv - 1 - i, j, nu - 1 - j)
            if edge == 0 and rng.random() < 0.35:
                keep[i, j] = False
    out = np.array(pts_out)
    inn = np.array(pts_in)
    v = np.vstack([out, inn])
    f = []
    off = nu * nv
    for i in range(nv - 1):
        for j in range(nu - 1):
            if not (keep[i, j] and keep[i, j + 1] and keep[i + 1, j + 1] and keep[i + 1, j]):
                continue
            a, b, c, d = i * nu + j, i * nu + j + 1, (i + 1) * nu + j + 1, (i + 1) * nu + j
            f += [(a, c, b), (a, d, c), (a + off, b + off, c + off), (a + off, c + off, d + off)]
    return Mesh(v, np.array(f))


def shard_parts(seed: int) -> list:
    m = _shard_mesh(SHARD_ARC[0], SHARD_ARC[1], seed)
    return [("shard", two_sided(m.moved(-m.v.mean(axis=0))), "glass")]


def chip_parts(seed: int) -> list:
    rng = np.random.default_rng(seed)
    a, b = CHIP
    pts = np.array([(-a / 2, -b / 2), (a / 2 * (0.6 + 0.4 * rng.random()), -b / 2), (a / 2, b * (0.1 * rng.random())),
                    (a / 4, b / 2), (-a / 2 * (0.5 + 0.5 * rng.random()), b / 2 * (0.7 + 0.3 * rng.random()))])
    from usdkit import prism
    return [("chip", prism(pts - pts.mean(axis=0), -GLASS_T / 2, GLASS_T / 2), "glass")]


def _bin_walls() -> list[tuple[tuple, tuple]]:
    """(size, centre) of the bin's floor and four walls."""
    ox, oy = BIN_OUT
    w, h = BIN_WALL, BIN_H
    return [((ox, oy, BIN_FLOOR), (0, 0, BIN_FLOOR / 2)),
            ((ox, w, h), (0, oy / 2 - w / 2, h / 2)), ((ox, w, h), (0, -oy / 2 + w / 2, h / 2)),
            ((w, oy - 2 * w, h), (ox / 2 - w / 2, 0, h / 2)), ((w, oy - 2 * w, h), (-ox / 2 + w / 2, 0, h / 2))]


def two_sided(m: Mesh) -> Mesh:
    """The same surface seen from both sides (thin shells: glass, the basket's woven wall)."""
    return Mesh(np.vstack([m.v, m.v]), np.vstack([m.f, m.f[:, ::-1] + len(m.v)]))


def bin_parts() -> list:
    ox, oy = BIN_OUT
    body = join(*[box(s, c, bevel=0.001) for s, c in _bin_walls()])
    # the liner: a thin black sheet lining the inside and folded over the rim
    liner_in = box((BIN_IN[0] - 0.002, BIN_IN[1] - 0.002, 0.001), (0, 0, BIN_FLOOR + 0.0006), round_xy=0.015)
    fold = []
    for sx, sy, cx, cy in ((ox + 0.004, 0.006, 0, oy / 2 - 0.002), (ox + 0.004, 0.006, 0, -oy / 2 + 0.002),
                           (0.006, oy + 0.004, ox / 2 - 0.002, 0), (0.006, oy + 0.004, -ox / 2 + 0.002, 0)):
        fold.append(box((sx, sy, 0.03), (cx, cy, BIN_H - 0.012)))
    walls = []
    for sx, sy, cx, cy in ((BIN_IN[0], 0.001, 0, BIN_IN[1] / 2 - 0.001), (BIN_IN[0], 0.001, 0, -BIN_IN[1] / 2 + 0.001),
                           (0.001, BIN_IN[1], BIN_IN[0] / 2 - 0.001, 0), (0.001, BIN_IN[1], -BIN_IN[0] / 2 + 0.001, 0)):
        walls.append(box((sx, sy, BIN_H - BIN_FLOOR), (cx, cy, BIN_FLOOR + (BIN_H - BIN_FLOOR) / 2)))
    return [("body", body, "binbody"), ("liner", join(liner_in, *fold, *walls), "liner")]


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


def convex(name: str, points, root: str, material: str = "PhysicsMaterial", contact_offset: float = 0.002) -> str:
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


def cube(name: str, size, centre, root: str, material: str = "PhysicsMaterial", contact_offset: float = 0.002) -> str:
    """A box collider (a PhysX box, not rendered: purpose "guide")."""
    (cx, cy, cz), (sx, sy, sz) = centre, size
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


BASKET_WALL_SEGMENTS = 20


def basket_colliders(root: str) -> list[str]:
    """The floor as one hull, the flared wall as thin convex panels between outline points, and each handle as a box."""
    cols = []
    base = _basket_outline(0.0)
    floor = np.vstack([np.c_[base * 0.98, np.zeros(len(base))], np.c_[_basket_outline(BASKET_FLOOR) * 0.98, np.full(len(base), BASKET_FLOOR)]])
    cols.append(convex("c_floor", floor, root))
    n = BASKET_WALL_SEGMENTS
    lo = _oval(BASKET_BASE[0], BASKET_BASE[1], n)
    hi = _oval(BASKET_RIM[0], BASKET_RIM[1], n)
    for i in range(n):
        j = (i + 1) % n
        pts = []
        for o, z in ((lo, 0.0), (hi, BASKET_H)):
            for k in (i, j):
                p = o[k]
                inward = -p / np.linalg.norm(p)
                pts += [(*p, z), (*(p + inward * BASKET_WALL), z)]
        cols.append(convex(f"c_wall{i:02d}", np.array(pts), root))
    for tag, side in (("l", 1.0), ("r", -1.0)):
        y_bar = side * (BASKET_RIM[1] / 2 + HANDLE_OUT)
        cols.append(cube(f"c_handle_{tag}", (HANDLE_LEN + HANDLE_T, HANDLE_T, HANDLE_W), (0.0, y_bar, HANDLE_Z), root))
        for sx in (-1.0, 1.0):  # the strap's two arms back to the rim
            y0 = side * (BASKET_RIM[1] / 2 - 0.004)
            cols.append(cube(f"c_arm_{tag}{'p' if sx > 0 else 'n'}", (HANDLE_T, abs(y_bar - y0), HANDLE_W),
                             (sx * HANDLE_LEN / 2, (y0 + y_bar) / 2, HANDLE_Z), root))
    return cols


def roll_hull() -> np.ndarray:
    return roll_parts()[0][1].v


def stem_hulls() -> list[np.ndarray]:
    """Foot, stem and stub, one hull each."""
    return [m.v for m in _stem_components()]


def shard_hulls(seed: int) -> list[np.ndarray]:
    """A curved piece as three convex strips across its curve, so the hulls stay within about 1 mm of the shell."""
    v = shard_parts(seed)[0][1].v
    xs = np.quantile(v[:, 0], np.linspace(0.0, 1.0, 6))
    out = []
    for a, b in zip(xs[:-1], xs[1:]):
        sel = v[(v[:, 0] >= a - 1e-6) & (v[:, 0] <= b + 1e-6)]
        if len(sel) >= 4:
            out.append(sel)
    return out


def bin_colliders(root: str) -> list[str]:
    return [cube(f"c_bin{i}", s, c, root) for i, (s, c) in enumerate(_bin_walls())]


# -- USD ----------------------------------------------------------------------------------------------------------------


def _write(path: Path, root: str, parts: list, colliders: list[str], mass: float | None, friction: tuple[float, float], *,
           solver_iters: int = 16, max_depen: float = 0.5) -> None:
    used = sorted({p[2] for p in parts})
    looks = "".join(_material_usda(m, root, LOOKS[m]) for m in used)
    meshes = "".join(_mesh_usda(p[0], p[1], p[2], root, LOOKS[p[2]], 35.0) for p in parts)
    if mass is None:
        head, body = "", ""
    else:
        head = """ (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
)"""
        body = f"""
    float physics:mass = {mass:.6f}
    int physxRigidBody:solverPositionIterationCount = {solver_iters}
    int physxRigidBody:solverVelocityIterationCount = 4
    float physxRigidBody:maxDepenetrationVelocity = {max_depen}"""
    text = f"""#usda 1.0
(
    defaultPrim = "{root}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}"{head}
{{{body}
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
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


# Friction (static, dynamic). The station's table uses PhysX's default material (0.5, 0.5); PhysX averages the two.
WICKER_FRICTION = (0.6, 0.5)
CRUST_FRICTION = (0.5, 0.45)
GLASS_FRICTION = (0.4, 0.35)
BIN_FRICTION = (0.5, 0.45)
SHARD_SEEDS = (3, 5)
CHIP_SEEDS = (11, 13, 17)


def write_all() -> None:
    make_textures()
    a = HERE / "assets"
    _write(a / "basket.usda", "basket", basket_parts(), basket_colliders("basket"), BASKET_MASS, WICKER_FRICTION, solver_iters=32)
    _write(a / "roll.usda", "roll", roll_parts(), [convex("c_roll", roll_hull(), "roll")], ROLL_MASS, CRUST_FRICTION)
    _write(a / "stem.usda", "stem", stem_parts(), [convex(f"c_stem{i}", h, "stem") for i, h in enumerate(stem_hulls())],
           STEM_PIECE_MASS, GLASS_FRICTION)
    for k, seed in enumerate(SHARD_SEEDS):
        _write(a / f"shard{k}.usda", "shard", shard_parts(seed),
               [convex(f"c_shard{i}", h, "shard", contact_offset=0.0015) for i, h in enumerate(shard_hulls(seed))],
               SHARD_MASS, GLASS_FRICTION)
    for k, seed in enumerate(CHIP_SEEDS):
        _write(a / f"chip{k}.usda", "chip", chip_parts(seed),
               [convex("c_chip", chip_parts(seed)[0][1].v, "chip", contact_offset=0.0015)], CHIP_MASS, GLASS_FRICTION)
    # a rigid body in the file (the harness puts a contact sensor on every object); task.py makes it kinematic
    _write(a / "bin.usda", "bin", bin_parts(), bin_colliders("bin"), 1.0, BIN_FRICTION)
