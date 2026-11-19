"""The apple-picking practice rig: an artificial apple tree in a pot, two imitation apples on magnetic breakaway spurs, and
the picking bucket: meshes, materials and colliders, written as plain-text USD.

Every named or handled object is a mesh with a real object's shape and size. Each collides through separate simple shapes
kept out of the render (purpose "guide"): PhysX boxes where the part is a box, short convex hulls elsewhere.

Frames (metres, z up):
- tree: origin at the middle of the pot's base, on the table. The trunk rises along +z. Each printed spur points out of the
  trunk along its own azimuth (``SPURS``), level, with a disc magnet at its tip.
- apple: origin at the apple's centre of mass. +x points along the stem, from the apple to the spur; the stem's cap sits on
  the spur's magnet at ``ABSCISSION`` on the x axis (the seat).
- bucket: origin at the middle of the bucket's base, on the table. The foam pad lies on its floor.

What changes when the stem pulls out of the apple is shown by switching prims on and off (task.py does it): the apple's
``visual/stem`` and ``visual/cap`` go, its ``visual/socket`` and ``visual/socket_magnet`` show, and the tree's
``visual/stem_<k>`` and ``visual/cap_<k>`` (the stem left on the spur) show.

Pure Python and NumPy (PIL and SciPy for the textures and hulls).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from usdkit import Look, Mesh, _corner_normals, _fmt, _material_usda, _mesh_usda, cylinder, join, lathe, tube

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
TEXTURES = HERE / "textures"

# -- the apple: a ripe 'Gala'-type apple, 75 mm across, 180 g ------------------------------------------------------------

APPLE_D = 0.075
APPLE_MASS = 0.18
APPLE_TOP = 0.0335          # the shoulders round the stem cavity, from the centre along the stem axis
APPLE_BOTTOM = -0.0330      # the calyx end
CAVITY_BOTTOM = 0.0205      # where the stem leaves the fruit (the stem cavity is 13 mm deep)
WELL_R = 0.008             # actual open installation well, 16 mm across
SOCKET_H = 0.0167          # socket mouth recessed at CAVITY_BOTTOM
SOCKET_BOTTOM = CAVITY_BOTTOM - SOCKET_H
SOCKET_BORE_R = 0.0064
SOCKET_MAGNET_TOP = SOCKET_BOTTOM + 0.0015 + 0.0032
STEM_L = 0.020              # the printed stem's flexible part, 6 mm thick, from the cavity ...
STEM_D = 0.006
CAP_L = 0.004               # ... to its cap, 15 mm across, holding a 12.7 mm steel disc that sits on the spur's magnet
CAP_D = 0.015
DISC_D = 0.0127
ABSCISSION = np.array([CAVITY_BOTTOM + STEM_L + CAP_L, 0.0, 0.0])   # the seat (cap on magnet), in the apple's frame

# -- the tree: a columnar apple tree in a 12 L plastic pot ---------------------------------------------------------------

POT_R_BOT = 0.1175          # pot: 235 mm across the base, ...
POT_R_TOP = 0.150           # ... 300 mm across the rim, ...
POT_H = 0.220               # ... 220 mm tall
SOIL_Z = 0.200              # the potting mix is filled to 20 mm below the rim
TRUNK_D = 0.025             # the trunk, 25 mm thick at the soil, ...
TRUNK_TOP = 0.500           # ... to 500 mm above the table
SPUR_L = 0.025              # a printed fruiting spur: 25 mm long out of the trunk, ...
SPUR_D = 0.016              # ... 16 mm across at its tip, where a 12.7 mm disc magnet sits flush
TREE_MASS = 11.5            # kg: the pot (0.7 kg), 11.3 L of dry sand under bark mulch (10.3 kg), trunk, spurs and leaves (0.5 kg)
TREE_COM_Z = 0.100
# (azimuth of the spur, degrees from +x towards +y; height of the apple's centre above the table)
SPURS = {"a": (160.0, 0.310), "b": (250.0, 0.350)}
APPLE_FROM_AXIS = TRUNK_D / 2 + SPUR_L + float(ABSCISSION[0])   # trunk axis to the apple's centre

# -- the bucket: a plastic picking bucket with a foam pad -----------------------------------------------------------

BKT_R_TOP = 0.145           # outer radius at the rim (290 mm across), ...
BKT_R_BOT = 0.125           # ... at the base (250 mm), ...
BKT_H = 0.230               # ... 230 mm deep
BKT_WALL = 0.003
BKT_FLOOR = 0.005
PAD_T = 0.025               # a 25 mm foam pad on the floor
PAD_TOP = BKT_FLOOR + PAD_T
BKT_MASS = 0.75             # kg, with the pad

# -- materials ------------------------------------------------------------------------------------------------------------

LOOKS = {
    "apple": Look(texture="apple_skin.png", roughness=0.38),
    "apple_cavity": Look(color=(0.55, 0.52, 0.22), roughness=0.7),
    "stem": Look(color=(0.24, 0.16, 0.07), roughness=0.6),
    "steel": Look(color=(0.62, 0.63, 0.65), metallic=1.0, roughness=0.35),
    "socket": Look(color=(0.05, 0.05, 0.05), roughness=0.8),
    "bark": Look(texture="bark.png", roughness=0.85, tile=0.08),
    "spur": Look(color=(0.30, 0.21, 0.14), roughness=0.6),
    "wood": Look(color=(0.78, 0.68, 0.48), roughness=0.85),
    "leaf": Look(texture="leaf.png", roughness=0.45),
    "soil": Look(texture="soil.png", roughness=0.95, tile=0.12),
    "pot": Look(color=(0.035, 0.035, 0.035), roughness=0.55),
    "bucket": Look(color=(0.18, 0.36, 0.20), roughness=0.5),
    "bucket_in": Look(color=(0.15, 0.31, 0.17), roughness=0.55),
    "foam": Look(texture="foam.png", roughness=0.95, tile=0.10),
}

FRICTION = {"apple": (1.20, 1.00), "bark": (0.70, 0.60), "pot": (0.60, 0.50), "bucket": (0.40, 0.35), "foam": (0.90, 0.80)}


# -- textures ------------------------------------------------------------------------------------------------------------


def _smooth(shape, seed: int, cycles) -> np.ndarray:
    """Tileable smooth noise with about `cycles` features across each axis (unit standard deviation)."""
    rng = np.random.default_rng(seed)
    h, w = shape
    cy, cx = (cycles, cycles) if np.isscalar(cycles) else cycles
    spec = np.fft.fft2(rng.normal(0.0, 1.0, (h, w)))
    fy = np.fft.fftfreq(h)[:, None] * h / cy
    fx = np.fft.fftfreq(w)[None, :] * w / cx
    out = np.real(np.fft.ifft2(spec * np.exp(-(fx ** 2 + fy ** 2))))
    return out / out.std()


def _apple_skin(n: int = 512) -> np.ndarray:
    """'Gala': an orange-red blush over most of a yellow ground colour, with darker red stripes running pole to pole, small
    pale lenticels, and green-yellow russet in the stem cavity. Rows run from the calyx (top row) to the stem (bottom row);
    columns go once round the apple."""
    v, u = np.mgrid[0:n, 0:n] / n
    ground = np.array([0.95, 0.80, 0.34])
    blush = np.array([0.86, 0.33, 0.12])
    stripe = np.array([0.62, 0.08, 0.06])
    side = 0.5 + 0.5 * np.cos(2 * np.pi * (u - 0.35))              # the sunny side
    cover = np.clip(0.55 + 0.55 * side + 0.22 * _smooth((n, n), 1, (3, 4)), 0, 1)
    cover *= np.clip(np.sin(np.pi * np.clip(v * 1.02, 0, 1)) * 2.2, 0, 1)  # less blush right at the poles
    streaks = 0.5 + 0.5 * np.tanh(1.2 * _smooth((n, n), 2, (2.5, 45)) + 0.6)
    s = np.clip(streaks * cover, 0, 1)
    rgb = ground * (1 - cover[..., None]) + blush * cover[..., None]
    rgb = rgb * (1 - 0.65 * s[..., None]) + stripe * 0.65 * s[..., None]
    rgb *= (1.0 + 0.04 * _smooth((n, n), 3, 20))[..., None]
    rng = np.random.default_rng(4)
    for _ in range(500):                                              # lenticels
        y, x = rng.integers(0, n, 2)
        r = rng.uniform(0.6, 1.2)
        yy, xx = np.ogrid[-2:3, -2:3]
        m = (xx * xx + yy * yy) <= r * r
        ys, xs = (y + yy) % n, (x + xx) % n
        rgb[ys, xs] = rgb[ys, xs] * (1 - 0.45 * m[..., None]) + np.array([0.95, 0.86, 0.60]) * 0.45 * m[..., None]
    russet = np.clip((v - 0.86) / 0.08, 0, 1)[..., None] * (0.7 + 0.3 * np.clip(_smooth((n, n), 5, 12), -1, 1)[..., None])
    rgb = rgb * (1 - russet) + np.array([0.60, 0.55, 0.24]) * russet
    return rgb


def _bark(n: int = 512) -> np.ndarray:
    """Young apple bark: smooth grey-brown with a reddish tinge, fine vertical fissures and pale lenticels."""
    base = np.array([0.36, 0.28, 0.22])
    tone = 1.0 + 0.10 * _smooth((n, n), 11, (4, 40)) + 0.06 * _smooth((n, n), 12, 16)
    fiss = np.clip(_smooth((n, n), 13, (2, 30)) - 1.2, 0, None)
    rgb = base * tone[..., None] * (1 - 0.35 * np.clip(fiss, 0, 1))[..., None]
    rng = np.random.default_rng(14)
    for _ in range(260):
        y, x = rng.integers(0, n, 2)
        rgb[y % n, (x + np.arange(-3, 4)) % n] = rgb[y % n, (x + np.arange(-3, 4)) % n] * 0.4 + np.array([0.62, 0.55, 0.45]) * 0.6
    return rgb


def _soil(n: int = 512) -> np.ndarray:
    """Moist potting mix: dark brown peat and bark crumbs with white perlite grains."""
    tone = 0.75 + 0.18 * _smooth((n, n), 21, 60) + 0.12 * _smooth((n, n), 22, 12)
    rgb = np.array([0.20, 0.13, 0.08]) * tone[..., None]
    crumbs = _smooth((n, n), 23, 90) > 1.4
    rgb[crumbs] = rgb[crumbs] * 1.6
    rng = np.random.default_rng(24)
    for _ in range(420):
        y, x = rng.integers(0, n, 2)
        r = rng.uniform(1.0, 2.6)
        yy, xx = np.ogrid[-3:4, -3:4]
        m = (xx * xx + yy * yy) <= r * r
        ys, xs = (y + yy) % n, (x + xx) % n
        rgb[ys, xs] = rgb[ys, xs] * (1 - m[..., None]) + np.array([0.86, 0.85, 0.80]) * m[..., None]
    return rgb


def _foam(n: int = 512) -> np.ndarray:
    """Grey open-cell foam."""
    tone = 0.62 + 0.08 * _smooth((n, n), 31, 120) + 0.05 * _smooth((n, n), 32, 20)
    pores = np.clip(_smooth((n, n), 33, 150) - 1.0, 0, None)
    return np.array([0.62, 0.66, 0.70]) * (tone - 0.12 * pores)[..., None]


def _leaf(n: int = 256) -> np.ndarray:
    """An apple leaf's upper side: mid green, a paler midrib and veins. Columns run along the leaf (base to tip)."""
    v, u = np.mgrid[0:n, 0:n] / n
    rgb = np.array([0.16, 0.36, 0.09]) * (1.0 + 0.08 * _smooth((n, n), 41, 8))[..., None]
    mid = np.exp(-((v - 0.5) / 0.012) ** 2)
    veins = np.exp(-((np.mod((u * 9 - np.abs(v - 0.5) * 6), 1.0) - 0.5) / 0.05) ** 2) * (np.abs(v - 0.5) < 0.45)
    light = np.clip(mid + 0.35 * veins, 0, 1)[..., None]
    return rgb * (1 - light) + np.array([0.45, 0.58, 0.25]) * light


def make_textures() -> None:
    from PIL import Image

    TEXTURES.mkdir(exist_ok=True)
    for name, fn in (("apple_skin.png", _apple_skin), ("bark.png", _bark), ("soil.png", _soil), ("foam.png", _foam),
                     ("leaf.png", _leaf)):
        if not (TEXTURES / name).exists():
            Image.fromarray((np.clip(fn(), 0, 1) * 255).astype(np.uint8)).save(TEXTURES / name)


# -- meshes ---------------------------------------------------------------------------------------------------------------


def _turn_to(m: Mesh, axis: np.ndarray) -> Mesh:
    """Turn a mesh built along +z so that +z points along `axis`."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, a)
    s, c = np.linalg.norm(v), float(np.dot(z, a))
    if s < 1e-9:
        r = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / s
        r = np.eye(3) + s * k + (1 - c) * (k @ k)
    return Mesh(m.v @ r.T, m.f)


def apple_profile() -> list[tuple[float, float]]:
    """(radius, height along the stem axis) from the calyx basin to the bottom of the stem cavity."""
    pts = [(0.0, APPLE_BOTTOM + 0.004), (0.004, APPLE_BOTTOM + 0.0035), (0.008, APPLE_BOTTOM + 0.0008)]
    widest = max(math.sin(math.pi * k / 300) ** 0.78 * (0.90 + 0.10 * (1 - math.cos(math.pi * k / 300)) / 2) for k in range(301))
    for k in range(1, 30):
        phi = math.pi * k / 30
        r = APPLE_D / 2 * math.sin(phi) ** 0.78 * (0.90 + 0.10 * (1 - math.cos(phi)) / 2) / widest
        z = -0.0332 * math.cos(phi) if math.cos(phi) > 0 else -APPLE_TOP * math.cos(phi)
        if r > 0.0095:
            pts.append((r, z))
    pts += [(0.0105, APPLE_TOP - 0.0012), (WELL_R, APPLE_TOP),
            (WELL_R, SOCKET_BOTTOM), (0.0, SOCKET_BOTTOM)]
    return pts


def _lathe_uv(m: Mesh, z0: float, z1: float) -> np.ndarray:
    """Per-corner texture coordinates for a lathed mesh: u once round the axis, v from z0 to z1."""
    p = m.v[m.f]
    u = (np.arctan2(p[..., 1], p[..., 0]) / (2 * np.pi)) % 1.0
    for row in u:
        if row.max() - row.min() > 0.5:
            row[row < 0.5] += 1.0
    v = (p[..., 2] - z0) / (z1 - z0)
    return np.stack([u, 1.0 - v], axis=-1)


def stem_meshes() -> dict:
    """The printed stem in the apple's frame (+x from the apple to the spur): the flexible part and the cap (brown TPU), and
    the steel disc on the cap's face."""
    x0, x1 = CAVITY_BOTTOM, CAVITY_BOTTOM + STEM_L
    flex = tube([(x, 0.0, 0.0) for x in np.linspace(x0, x1, 6)], STEM_D / 2, 12)
    cap = lathe([(0.0, x1), (CAP_D/2,x1), (CAP_D/2,x1+CAP_L),
                 (0.00645,x1+CAP_L), (0.00645,x1+0.0009), (0.0,x1+0.0009)],24).turned(90.0,"y")
    peg = cylinder(0.0063, 0.0089, (CAVITY_BOTTOM-0.0089/2,0,0),axis="x",segments=24)
    disc = cylinder(DISC_D/2,0.003,(ABSCISSION[0]-.0015,0,0),axis="x",segments=24)
    end = cylinder(DISC_D/2,0.003,(CAVITY_BOTTOM-.0105,0,0),axis="x",segments=24)
    return {"stem": join(flex, cap, peg), "cap": join(disc,end)}


def apple_meshes() -> dict:
    """In the apple's frame (+x along the stem): the fruit with per-corner uv, the printed stem, and the stem's socket as
    it shows once the stem has pulled out (a dark bore with the magnet at its bottom)."""
    body = lathe(apple_profile(), 40)
    uv = _lathe_uv(body, APPLE_BOTTOM, APPLE_TOP)
    socket = lathe([(0,SOCKET_BOTTOM),(WELL_R,SOCKET_BOTTOM),(WELL_R,CAVITY_BOTTOM),
                    (SOCKET_BORE_R,CAVITY_BOTTOM),(SOCKET_BORE_R,SOCKET_BOTTOM+.0015),(0,SOCKET_BOTTOM+.0015)],32)
    magnet = cylinder(DISC_D/2,0.003175,(0,0,SOCKET_BOTTOM+.0015+.003175/2),segments=32)
    turn = lambda m: m.turned(90.0, "y")  # noqa: E731  (+z -> +x)
    out = {"apple": (turn(body), uv), "socket": turn(socket), "socket_magnet": turn(magnet)}
    out.update(stem_meshes())
    return out


def apple_hull_points() -> np.ndarray:
    """The fruit's outside (no cavities), for its collider: at most about 2 mm from the visual skin."""
    prof = [(r, z) for r, z in apple_profile() if r > 0.009]
    pts = [(0.0, 0.0, APPLE_BOTTOM + 0.0008), (0.0, 0.0, APPLE_TOP - 0.0012)]
    for r, z in prof[:: max(1, len(prof) // 9)]:
        for k in range(12):
            t = 2 * math.pi * k / 12 + (0.13 if len(pts) % 2 else 0.0)
            pts.append((r * math.cos(t), r * math.sin(t), z))
    p = np.array(pts)
    return p[:, [2, 1, 0]] * np.array([1.0, 1.0, -1.0])  # turned so +z goes to +x (same turn as apple_meshes)


def spur_mesh(length: float = SPUR_L) -> tuple[Mesh, Mesh]:
    """A printed fruiting spur along +x from x = 0 (the trunk's surface) to `length`, 19 mm across at its base and 16 mm
    at its tip; and the disc magnet's face, flush with the tip."""
    body = lathe([(0.0, -0.004), (0.0095, -0.004), (SPUR_D / 2, length - 0.0008), (SPUR_D / 2 - 0.0008, length), (0.0, length)], 32)
    magnet = cylinder(DISC_D / 2, 0.0004, (0.0, 0.0, length - 0.0001), segments=32)
    return body.turned(90.0, "y"), magnet.turned(90.0, "y")


def leaf_mesh(length: float, width: float) -> tuple[Mesh, np.ndarray]:
    """An apple leaf along +x from its base at the origin, upper side +z, slightly cupped: two faces back to back."""
    nx, ny = 10, 5
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(-1, 1, ny)
    verts, uvs = [], []
    for x in xs:
        half = width / 2 * (math.sin(math.pi * min(1.0, x * 1.15)) ** 0.8 if x < 0.87 else max(0.0, (1 - x) / 0.13) * 0.45)
        for y in ys:
            verts.append((x * length, y * half, 0.006 * (y * y) * math.sin(math.pi * x) - 0.004 * x * x))
            uvs.append((x, 0.5 + 0.5 * y))
    v = np.array(verts)
    f = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = i * ny + j, i * ny + j + 1, (i + 1) * ny + j + 1, (i + 1) * ny + j
            f += [(a, d, c), (a, c, b)]
    f = np.array(f)
    back = Mesh(v - np.array([0, 0, 0.0004]), f[:, ::-1])
    mesh = Mesh(np.vstack([v, back.v]), np.vstack([f, back.f + len(v)]))
    uv = np.vstack([np.array(uvs), np.array(uvs)])[mesh.f]
    return mesh, uv


def _place(m: Mesh, rot: np.ndarray, t) -> Mesh:
    return Mesh(m.v @ rot.T + np.asarray(t, float), m.f)


def _rz(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _ry(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rx(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def spur_frame(k: str) -> tuple[np.ndarray, np.ndarray]:
    """A spur's base on the trunk's axis and its outward direction, in the tree's frame."""
    az, h = SPURS[k]
    d = np.array([math.cos(math.radians(az)), math.sin(math.radians(az)), 0.0])
    return np.array([0.0, 0.0, h]), d


def apple_rest(k: str) -> tuple[np.ndarray, np.ndarray]:
    """Apple k's centre and rotation (apple frame -> tree frame) as it hangs on the tree."""
    base, d = spur_frame(k)
    centre = base + APPLE_FROM_AXIS * d
    rot = _rz(SPURS[k][0] + 180.0)   # apple +x (towards the stem) points back at the trunk
    return centre, rot


def abscission_in_tree(k: str) -> np.ndarray:
    centre, rot = apple_rest(k)
    return centre + rot @ ABSCISSION


def tree_meshes() -> list:
    parts = []
    pot = lathe([(0.0, 0.0), (POT_R_BOT - 0.004, 0.0), (POT_R_BOT, 0.004), (POT_R_TOP - 0.004, POT_H - 0.018),
                 (POT_R_TOP, POT_H - 0.016), (POT_R_TOP + 0.002, POT_H - 0.006), (POT_R_TOP, POT_H),
                 (POT_R_TOP - 0.006, POT_H), (POT_R_TOP - 0.008, POT_H - 0.010), (POT_R_TOP - 0.010, SOIL_Z - 0.01),
                 (0.0, SOIL_Z - 0.01)], 48)
    parts.append(("pot", pot, "pot"))
    soil = lathe([(0.0, SOIL_Z - 0.012), (POT_R_TOP - 0.009, SOIL_Z - 0.012), (POT_R_TOP - 0.009, SOIL_Z - 0.001),
                  (0.06, SOIL_Z + 0.002), (TRUNK_D / 2 + 0.004, SOIL_Z + 0.005), (0.0, SOIL_Z + 0.006)], 40)
    parts.append(("soil", soil, "soil"))
    trunk = cylinder(TRUNK_D / 2, TRUNK_TOP - SOIL_Z + 0.01, (0.0, 0.0, (TRUNK_TOP + SOIL_Z - 0.01) / 2), segments=20)
    parts.append(("trunk", trunk, "bark"))   # a 25 mm dowel, bark-brown
    tip = lathe([(TRUNK_D / 2, 0.0), (0.009, 0.02), (0.003, 0.05), (0.0, 0.058)], 14).moved((0.0, 0.0, TRUNK_TOP - 0.004))
    parts.append(("leader", tip, "bark"))
    for k in SPURS:
        base, d = spur_frame(k)
        rot = _rz(SPURS[k][0])
        spur, magnet = spur_mesh()
        start = base + d * (TRUNK_D / 2)
        parts.append((f"spur_{k}", _place(spur, rot, start), "spur"))
        parts.append((f"magnet_{k}", _place(magnet, rot, start), "steel"))
        centre, arot = apple_rest(k)
        sm = stem_meshes()   # shown on the spur after a stem pull
        parts.append((f"stem_{k}", _place(sm["stem"], arot, centre), "stem", True))
        parts.append((f"cap_{k}", _place(sm["cap"], arot, centre), "steel", True))
    for i, (az, z, tilt, length) in enumerate(leaf_layout()):
        mesh, uv = leaf_mesh(length, length * 0.48)
        rot = _rz(az) @ _ry(-tilt)
        r = TRUNK_D / 2 * (1.0 - 0.36 * (z - SOIL_Z) / (TRUNK_TOP - SOIL_Z))
        petiole = tube([(0.0, 0.0, 0.0), (0.012, 0.0, -0.002), (0.022, 0.0, -0.001)], 0.0009, 6)
        leaf = join(petiole, mesh.moved((0.022, 0.0, -0.001)))
        uvs = np.vstack([np.full((len(petiole.f), 3, 2), 0.02), uv])
        t = np.array([r * math.cos(math.radians(az)), r * math.sin(math.radians(az)), z])
        parts.append((f"leaf{i:02d}", (_place(leaf, rot, t), uvs), "leaf"))
    return parts


def leaf_layout() -> list[tuple[float, float, float, float]]:
    """(azimuth, height, tilt up from level, length) of each leaf: round the trunk, clear of the apples and of the arm's
    way to them (the side facing the right arm, between and around the two apples)."""
    rng = np.random.default_rng(7)
    out = []
    z = SOIL_Z + 0.05
    while z < TRUNK_TOP - 0.005:
        az = float(rng.uniform(0, 360))
        clear = all(not (abs(((az - a + 180) % 360) - 180) < 55 and abs(z - h) < 0.075) for a, h in SPURS.values())
        clear = clear and not (180 < az < 260 and min(abs(z - h) for _, h in SPURS.values()) < 0.10)
        if clear:
            out.append((az, z, float(rng.uniform(15, 50)), float(rng.uniform(0.055, 0.075))))
        z += float(rng.uniform(0.004, 0.008))
    for az in (20, 140, 260):  # a tuft at the top
        out.append((float(az), TRUNK_TOP - 0.01, 55.0, 0.06))
    return out


def bucket_meshes() -> list:
    t, f = BKT_WALL, BKT_FLOOR
    outer = [(0.0, 0.0), (BKT_R_BOT - 0.006, 0.0), (BKT_R_BOT, 0.006), (BKT_R_TOP, BKT_H - 0.012), (BKT_R_TOP + 0.006, BKT_H - 0.010),
             (BKT_R_TOP + 0.007, BKT_H - 0.003), (BKT_R_TOP + 0.002, BKT_H), (BKT_R_TOP - t, BKT_H)]
    inner = [(BKT_R_TOP - t, BKT_H - 0.004), (BKT_R_BOT - t + 0.0005, f + 0.004), (BKT_R_BOT - t - 0.004, f), (0.0, f)]
    shell = lathe(outer + inner, 56)
    ribs = []   # two moulded hoops round the outside (open bands: nothing across the inside)
    for z in (0.06, 0.12):
        r = BKT_R_BOT + (BKT_R_TOP - BKT_R_BOT) * z / BKT_H
        ribs.append(lathe([(r - 0.0005, z - 0.003), (r + 0.0018, z - 0.0015), (r + 0.0018, z + 0.0015), (r - 0.0005, z + 0.003)], 56))
    pad = lathe([(0.0, f), (BKT_R_BOT - t - 0.006, f), (BKT_R_BOT - t - 0.003, f + 0.004), (BKT_R_BOT - t - 0.002, PAD_TOP - 0.004),
                 (BKT_R_BOT - t - 0.005, PAD_TOP), (0.0, PAD_TOP)], 48)
    handle_pts = [(BKT_R_TOP * math.cos(a), BKT_R_TOP * math.sin(a), BKT_H - 0.03 - 0.02 * math.sin(a)) for a in np.linspace(0.1, math.pi - 0.1, 24)]
    handle = tube(handle_pts, 0.0028, 8)
    lugs = [cylinder(0.009, 0.006, (0.0, s * (BKT_R_TOP + 0.003), BKT_H - 0.03), axis="y", segments=12) for s in (-1, 1)]
    return [("bucket", join(shell, *ribs), "bucket"), ("pad", pad, "foam"), ("handle", join(handle, *lugs), "pot")]


# -- colliders ------------------------------------------------------------------------------------------------------------


def _hull_faces(points: np.ndarray) -> np.ndarray:
    from scipy.spatial import ConvexHull

    hull = ConvexHull(points)
    centre = points.mean(axis=0)
    faces = []
    for tri in hull.simplices:
        a, b, c = points[tri]
        if np.dot(np.cross(b - a, c - a), a - centre) < 0:
            tri = tri[[0, 2, 1]]
        faces.append(tri)
    return np.array(faces)


def _material(name: str, friction: tuple[float, float]) -> str:
    return f"""
    def Material "{name}" (
        prepend apiSchemas = ["PhysicsMaterialAPI", "PhysxMaterialAPI"]
    )
    {{
        float physics:staticFriction = {friction[0]:.4f}
        float physics:dynamicFriction = {friction[1]:.4f}
        float physics:restitution = 0
    }}"""


def _cube(name: str, lo, hi, root: str, material: str, contact_offset: float = 0.003) -> str:
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


def _hull(name: str, points, root: str, material: str, contact_offset: float = 0.003) -> str:
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


def _ring(r: float, z: float, n: int, phase: float = 0.0) -> np.ndarray:
    a = np.linspace(0, 2 * np.pi, n, endpoint=False) + phase
    return np.c_[r * np.cos(a), r * np.sin(a), np.full(n, z)]


def tree_colliders(root: str) -> list[str]:
    # The filled body ends at soil level. Separate convex rim sectors keep
    # the opening above the soil hollow and support apples on the real lip.
    pot = np.vstack([_ring(POT_R_BOT, 0.0, 24), _ring(POT_R_TOP - .004, SOIL_Z, 24)])
    out = [_hull("c_pot", pot, root, "PotMaterial")]
    segments = 24
    for i in range(segments):
        pts = [(r * math.cos(a), r * math.sin(a), z)
               for a in (2 * math.pi * i / segments, 2 * math.pi * (i + 1) / segments)
               for r, z in ((POT_R_TOP - .010, SOIL_Z), (POT_R_TOP - .004, SOIL_Z),
                            (POT_R_TOP - .006, POT_H), (POT_R_TOP, POT_H))]
        out.append(_hull(f"c_rim_{i:02}", np.asarray(pts), root, "PotMaterial"))
    h = TRUNK_D * 0.42
    out.append(_cube("c_trunk", (-h, -h, SOIL_Z), (h, h, TRUNK_TOP - 0.02), root, "BarkMaterial"))
    for k in SPURS:
        base,d=spur_frame(k);R=_rz(SPURS[k][0]);start=base+d*(TRUNK_D/2)
        rings=np.vstack([_ring(.0095,-.004,16),_ring(SPUR_D/2,SPUR_L,16)])
        out.append(_hull("c_spur_"+k,rings[:,[2,1,0]]@R.T+start,root,"BarkMaterial",.002))
    return out


def apple_colliders(root: str) -> list[str]:
    """Real open installation well: bulk fruit plus finite annular crown sectors.

    The socket/magnet belong to the apple rigid body. Above the magnet there is
    a12.8mm bore up to the recessed socket mouth, then a16mm open counterbore.
    No single convex hull bridges or plugs that opening.
    """
    profile=apple_profile()[:-2]  # outer skin, including the mouth rim
    def radius(x):
        values=[]
        for (ra,xa),(rb,xb) in zip(profile[:-1],profile[1:]):
            if min(xa,xb)<=x<=max(xa,xb) and abs(xb-xa)>1e-12:values.append(ra+(rb-ra)*(x-xa)/(xb-xa))
        return max(values)
    turn=lambda p:p[:,[2,1,0]]*np.array([1.,1.,-1.])
    bulk=[_ring(r,x,16)for r,x in profile if x<=SOCKET_MAGNET_TOP]
    bulk.append(_ring(radius(SOCKET_MAGNET_TOP),SOCKET_MAGNET_TOP,16))
    out=[_hull("c_apple_bulk",turn(np.vstack(bulk)),root,"AppleMaterial",.002)]
    for band,(lo,hi,inner)in enumerate([(SOCKET_MAGNET_TOP,CAVITY_BOTTOM,SOCKET_BORE_R),(CAVITY_BOTTOM,APPLE_TOP-.0012,WELL_R)]):
        stations=sorted(set([lo,hi]+[x for _,x in profile if lo<x<hi]))
        for i in range(16):
            a0,a1=2*math.pi*i/16,2*math.pi*(i+1)/16;v=[]
            for a in (a0,a1):
                v += [(radius(x)*math.cos(a),radius(x)*math.sin(a),x)for x in stations]
                v += [(inner/math.cos(math.pi/16)*math.cos(a),inner/math.cos(math.pi/16)*math.sin(a),x)for x in (lo,hi)]
            out.append(_hull(f"c_apple_well_{band}_{i:02}",turn(np.array(v)),root,"AppleMaterial",.002))
    return out


def bucket_colliders(root: str) -> list[str]:
    out = [_hull("c_floor", np.vstack([_ring(BKT_R_BOT, 0.0, 24), _ring(BKT_R_BOT, BKT_FLOOR, 24)]), root, "BucketMaterial")]
    out.append(_hull("c_pad", np.vstack([_ring(BKT_R_BOT - BKT_WALL - 0.003, BKT_FLOOR, 24),
                                         _ring(BKT_R_BOT - BKT_WALL - 0.003, PAD_TOP, 24)]), root, "FoamMaterial", 0.002))
    n = 18
    for i in range(n):
        a0, a1 = 2 * np.pi * i / n, 2 * np.pi * (i + 1) / n
        pts = []
        for a in (a0 - 0.01, a1 + 0.01):
            for z in (0.0, BKT_H):
                r_in = BKT_R_BOT - BKT_WALL + (BKT_R_TOP - BKT_R_BOT) * z / BKT_H
                for r in (r_in, r_in + 0.010):
                    pts.append((r * math.cos(a), r * math.sin(a), z))
        out.append(_hull(f"c_wall{i:02d}", np.array(pts), root, "BucketMaterial"))
    return out


# -- writing --------------------------------------------------------------------------------------------------------------


def _mesh_uv_usda(name: str, mesh: Mesh, uv: np.ndarray, mat: str, root: str, crease: float, hidden: bool) -> str:
    v, f = mesh.v, mesh.f
    normals = _corner_normals(v, f, crease).reshape(-1, 3)
    vis = '\n            token visibility = "invisible"' if hidden else ""
    return f"""
        def Mesh "{name}" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {{{vis}
            rel material:binding = </{root}/Looks/{mat}>
            uniform bool doubleSided = 0
            int[] faceVertexCounts = [{", ".join("3" for _ in range(len(f)))}]
            int[] faceVertexIndices = [{", ".join(str(int(i)) for i in f.reshape(-1))}]
            point3f[] points = [{_fmt(v)}]
            normal3f[] normals = [{_fmt(normals, 4)}] (
                interpolation = "faceVarying"
            )
            texCoord2f[] primvars:st = [{_fmt(uv.reshape(-1, 2), 4)}] (
                interpolation = "faceVarying"
            )
            uniform token subdivisionScheme = "none"
        }}"""


def _write(path: Path, root: str, parts: list, colliders: list[str], materials: str, *, mass: float | None,
           com=None, solver_iters: int = 16, contact_report: bool = False) -> None:
    used = sorted({p[2] for p in parts})
    looks = "".join(_material_usda(m, root, LOOKS[m]) for m in used)
    meshes = []
    for p in parts:
        name, mesh, look = p[0], p[1], p[2]
        hidden = len(p) > 3 and p[3]
        if isinstance(mesh, tuple):
            meshes.append(_mesh_uv_usda(name, mesh[0], mesh[1], look, root, 40.0, hidden))
        else:
            meshes.append(_mesh_usda(name, mesh, look, root, LOOKS[look], 40.0, hidden=hidden))
    body = ""
    apis = '"PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"'
    report = ""
    if contact_report:
        apis += ', "PhysxContactReportAPI"'
        report = "\n    float physxContactReport:threshold = 0"
    if mass is not None:
        centre = "" if com is None else f"\n    point3f physics:centerOfMass = ({com[0]:.5f}, {com[1]:.5f}, {com[2]:.5f})"
        body = f"""
    float physics:mass = {mass:.5f}{centre}
    int physxRigidBody:solverPositionIterationCount = {solver_iters}
    int physxRigidBody:solverVelocityIterationCount = 4
    float physxRigidBody:maxDepenetrationVelocity = 0.5{report}
"""
    text = f"""#usda 1.0
(
    defaultPrim = "{root}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}" (
    prepend apiSchemas = [{apis}]
)
{{{body}{materials}
{"".join(colliders)}

    def Scope "Looks"
    {{{looks}
    }}

    def Xform "visual"
    {{{"".join(meshes)}
    }}
}}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def _stamp() -> str:
    import hashlib

    h = hashlib.sha256()
    for name in ("scene_assets.py", "usdkit.py"):
        h.update((HERE / name).read_bytes())
    return h.hexdigest()[:16]


def write_all() -> None:
    """Write the textures and the USD files, unless they were already written by this version of the code."""
    stamp = ASSETS / ".stamp"
    if stamp.exists() and stamp.read_text() == _stamp() and all((ASSETS / f"{n}.usda").exists() for n in ("tree", "bucket")):
        return
    make_textures()
    root = "tree"
    mats = _material("PotMaterial", FRICTION["pot"]) + _material("BarkMaterial", FRICTION["bark"])
    _write(ASSETS / "tree.usda", root, tree_meshes(), tree_colliders(root), mats, mass=TREE_MASS, com=(0.0, 0.0, TREE_COM_Z))

    am = apple_meshes()
    for k in SPURS:
        root = f"apple_{k}"
        parts = [("apple", am["apple"], "apple"), ("stem", am["stem"], "stem"), ("cap", am["cap"], "steel"),
                 ("socket", am["socket"], "socket"), ("socket_magnet", am["socket_magnet"], "steel")]
        cols = apple_colliders(root)
        _write(ASSETS / f"{root}.usda", root, parts, cols, _material("AppleMaterial", FRICTION["apple"]), mass=APPLE_MASS,
               com=(0.0, 0.0, 0.0), solver_iters=32, contact_report=True)

    root = "bucket"
    mats = _material("BucketMaterial", FRICTION["bucket"]) + _material("FoamMaterial", FRICTION["foam"])
    _write(ASSETS / "bucket.usda", root, bucket_meshes(), bucket_colliders(root), mats, mass=BKT_MASS, com=(0.0, 0.0, 0.03))
    stamp.write_text(_stamp())


if __name__ == "__main__":
    write_all()
