"""The scene's models, written as plain-text USD when the task file loads (no pxr needed).

A white ceramic sink with a chrome tap and water standing in it, a tiled splashback with a wall socket, a folded towel, and
a hair dryer whose body, cord and plug form one floating articulation: the dryer's body is the root link, and the cord is a
chain of short capsules joined by revolute joints with alternating axes, so it bends any way but does not twist. Its joints
have damping and no stiffness, so the cord keeps no memory of its starting shape.

All sizes in metres. `LAYOUT` positions are relative to the table top.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import usdkit as uk
from usdkit import Look, Mesh, box, collider, cylinder, join, lathe, prism, tube, write_usda

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"

# -- layout ------------------------------------------------------------------------------------------------------------

SINK_XY = (0.31, -0.03)
SINK_SIZE = (0.30, 0.26)  # outer, x and y
SINK_H = 0.08
SINK_WALL = 0.015
SINK_FLOOR = 0.015
WATER_LEVEL = 0.065  # 5 cm of water above the sink's floor

BARREL_R = 0.032
BARREL_L = 0.20
HANDLE = (-0.145, -0.03)  # along the body's x, from its end to where it enters the barrel
HANDLE_Y, HANDLE_Z = 0.034, 0.030  # thickness across the grip, and height as it lies
BOOT_END = -0.165  # the cord leaves here
DRYER_XY = (0.375, -0.03)  # the middle of the barrel
DRYER_MASS = 0.35
DRYER_COM = (-0.03, 0.0, 0.0)  # at the top of the handle, as in a real dryer: the motor sits there
DRYER_TILT = -math.asin((BARREL_R - HANDLE_Z / 2) / abs(HANDLE[0]))  # about y: barrel and handle's end both on the floor
DRYER_Z = SINK_FLOOR + BARREL_R

SOCKET = (0.29, 0.24, 0.09)  # middle of the socket on the plate's front face
PLUG_SIZE = (0.050, 0.032, 0.022)
PIN_BLOCK = (0.020, 0.024, 0.006)  # the collider standing for the two pins
PLUG_MID = (SOCKET[0] - PLUG_SIZE[0] / 2, SOCKET[1], SOCKET[2])
WALL_SIZE = (0.042, 0.20, 0.14)

TOWEL_XY = (0.26, -0.33)
TOWEL_SIZE = (0.30, 0.24, 0.03)

CORD_R = 0.0035
LINK_L = 0.02
LINK_MASS = 0.006  # lighter links in a pinch blow up the solver at 240 Hz
PLUG_MASS = 0.04

# The cord leaves the handle's end, climbs the sink's front wall, crosses the rim and drops to the table, runs along the
# table with some slack, and rises to the back of the plug. Control points, x, y, height above the table.
CORD_POINTS = (
    (0.199, -0.03, 0.040), (0.1875, -0.03, 0.062), (0.1835, -0.03, 0.082), (0.1675, -0.03, 0.0935), (0.1515, -0.03, 0.082),
    (0.1485, -0.03, 0.040), (0.143, -0.024, 0.0065), (0.11, 0.0, 0.0055), (0.08, 0.05, 0.0055), (0.075, 0.11, 0.0055),
    (0.105, 0.155, 0.0055), (0.16, 0.165, 0.0055), (0.195, 0.185, 0.0055), (0.210, 0.222, 0.0065), (0.211, 0.238, 0.040),
    (0.215, 0.24, 0.075), (0.227, 0.24, 0.089),
)


# -- rotations and the chain -------------------------------------------------------------------------------------------


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def dryer_pose(top: float) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """The body's starting position and orientation (x, y, z, w)."""
    pos = (DRYER_XY[0], DRYER_XY[1], top + DRYER_Z)
    return pos, (0.0, math.sin(DRYER_TILT / 2), 0.0, math.cos(DRYER_TILT / 2))


def _catmull_rom(points: np.ndarray, samples: int = 24) -> np.ndarray:
    p = np.vstack([2 * points[0] - points[1], points, 2 * points[-1] - points[-2]])
    out = [points[0]]
    for i in range(1, len(p) - 2):
        for t in np.linspace(0.0, 1.0, samples + 1)[1:]:
            t2, t3 = t * t, t * t * t
            out.append(0.5 * (2 * p[i] + (p[i + 1] - p[i - 1]) * t + (2 * p[i - 1] - 5 * p[i] + 4 * p[i + 1] - p[i + 2]) * t2
                              + (3 * p[i] - p[i - 1] - 3 * p[i + 1] + p[i + 2]) * t3))
    return np.array(out)


def cord_world_points() -> np.ndarray:
    """Joint points of the cord as authored, in table-top coordinates: the first at the dryer's boot, the last at the plug.
    Links are LINK_L long, except the last, which takes up what remains (up to twice that)."""
    r = rot_y(DRYER_TILT)
    start = np.array([DRYER_XY[0], DRYER_XY[1], DRYER_Z]) + r @ np.array([BOOT_END, 0.0, 0.0])
    end = np.array(PLUG_MID) - np.array([PLUG_SIZE[0] / 2, 0.0, 0.0])
    path = _catmull_rom(np.vstack([start, np.array(CORD_POINTS), end]))
    pts = [path[0]]
    i = 1
    while i < len(path):
        while i < len(path) and np.linalg.norm(path[i] - pts[-1]) < LINK_L:
            i += 1
        if i < len(path):
            a, b = path[i - 1], path[i]
            lo, hi = 0.0, 1.0
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                if np.linalg.norm(a + mid * (b - a) - pts[-1]) < LINK_L:
                    lo = mid
                else:
                    hi = mid
            pts.append(a + lo * (b - a))
    if np.linalg.norm(end - pts[-1]) < 0.25 * LINK_L:
        pts.pop()
    pts.append(end)
    return np.array(pts)


def _rot_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    c = float(u @ v)
    axis = np.cross(u, v)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return np.eye(3)
    k = axis / s
    kk = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    ang = np.arctan2(s, c)
    return np.eye(3) + np.sin(ang) * kk + (1 - np.cos(ang)) * kk @ kk


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


def _chain_frames(points: np.ndarray) -> list[np.ndarray]:
    """Each link's rotation (columns: x along the link, y, z), carried along the chain so the links do not roll."""
    dirs = np.diff(points, axis=0)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    x = dirs[0]
    up = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    y = np.cross(up, x)
    y /= np.linalg.norm(y)
    frames = [np.column_stack([x, y, np.cross(x, y)])]
    for i in range(1, len(dirs)):
        frames.append(_rot_between(dirs[i - 1], dirs[i]) @ frames[-1])
    return frames


def _fmt(v) -> str:
    return "(" + ", ".join(f"{float(a):.7f}" for a in v) + ")"


# -- looks -------------------------------------------------------------------------------------------------------------

LOOKS = {
    "ceramic": Look(color=(0.86, 0.86, 0.84), roughness=0.18),
    "chrome": Look(color=(0.82, 0.83, 0.85), metallic=1.0, roughness=0.16),
    "water": Look(color=(0.42, 0.68, 0.82), roughness=0.05, opacity=0.38),
    "dark": Look(color=(0.015, 0.015, 0.018), roughness=0.6),
    "tile": Look(color=(1.0, 1.0, 1.0), roughness=0.25, texture="tile.png", tile=0.20),
    "plate": Look(color=(0.84, 0.84, 0.82), roughness=0.35),
    "plate_ring": Look(color=(0.66, 0.66, 0.65), roughness=0.4),
    "terry": Look(color=(1.0, 1.0, 1.0), roughness=0.95, texture="terry.png", tile=0.10),
    "body": Look(color=(0.62, 0.05, 0.07), roughness=0.32),
    "body_dark": Look(color=(0.02, 0.02, 0.022), roughness=0.45),
    "switch": Look(color=(0.82, 0.82, 0.80), roughness=0.4),
    "rubber": Look(color=(0.012, 0.012, 0.014), roughness=0.55),
    "brass": Look(color=(0.78, 0.62, 0.30), metallic=1.0, roughness=0.3),
}


def _make_textures() -> None:
    """Textures made from noise: grouted wall tiles and a terry towel's loops."""
    from PIL import Image

    folder = HERE / "textures"
    folder.mkdir(exist_ok=True)
    if not (folder / "tile.png").exists():
        n = 512
        rng = np.random.default_rng(3)
        img = np.full((n, n, 3), 0.90)
        img += rng.normal(0.0, 0.004, (n, n, 1))
        tile = n // 2  # two tiles per 20 cm: 10 cm tiles
        grout = 6
        for k in (0, tile):
            img[k:k + grout, :] = (0.62, 0.62, 0.60)
            img[:, k:k + grout] = (0.62, 0.62, 0.60)
        Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(folder / "tile.png")
    if not (folder / "terry.png").exists():
        n = 512
        rng = np.random.default_rng(5)
        f = rng.normal(0.0, 1.0, (n, n))
        spectrum = np.fft.fft2(f)
        fx, fy = np.fft.fftfreq(n)[None, :], np.fft.fftfreq(n)[:, None]
        band = np.exp(-((np.hypot(fx, fy) - 0.10) / 0.05) ** 2)
        loops = np.real(np.fft.ifft2(spectrum * band))
        loops = np.clip(loops / (2.2 * loops.std()), -1.0, 1.0)
        base = np.array([0.20, 0.40, 0.58])
        img = base[None, None, :] * (0.92 + 0.30 * loops[:, :, None])
        Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(folder / "terry.png")


# -- the sink, the wall and the towel ----------------------------------------------------------------------------------


def _rings(rings: list[tuple[np.ndarray, float]]) -> Mesh:
    """A closed surface through outlines of equal length, from the bottom's outer edge round to the middle of the last."""
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


def _make_water_clear(path: str, root: str, name: str) -> None:
    """Replaces a material with tinted clear glass: a preview surface's opacity renders opaque under this renderer's settings."""
    text = Path(path).read_text()
    start = text.index(f'def Material "{name}"')
    depth, i = 0, text.index("{", start)
    for j in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            break
    base = f"/{root}/Looks/{name}"
    glass = f"""def Material "{name}"
        {{
            token outputs:mdl:surface.connect = <{base}/Shader.outputs:out>
            token outputs:mdl:displacement.connect = <{base}/Shader.outputs:out>
            token outputs:mdl:volume.connect = <{base}/Shader.outputs:out>

            def Shader "Shader"
            {{
                uniform token info:implementationSource = "sourceAsset"
                uniform asset info:mdl:sourceAsset = @OmniGlass.mdl@
                uniform token info:mdl:sourceAsset:subIdentifier = "OmniGlass"
                color3f inputs:glass_color = (0.70, 0.88, 0.96)
                float inputs:glass_ior = 1.05
                float inputs:frosting_roughness = 0
                bool inputs:thin_walled = 1
                token outputs:out
            }}
        }}"""
    Path(path).write_text(text[:start] + glass + text[j + 1:])


def write_sink() -> str:
    sx, sy = SINK_SIZE
    ix, iy = sx - 2 * SINK_WALL, sy - 2 * SINK_WALL
    o = uk._rounded_outline
    seg = 6
    basin = _rings([
        (o(sx, sy, 0.030, seg), 0.0),
        (o(sx, sy, 0.030, seg), SINK_H - 0.004),
        (o(sx - 0.008, sy - 0.008, 0.026, seg), SINK_H),
        (o(ix + 0.008, iy + 0.008, 0.024, seg), SINK_H),
        (o(ix, iy, 0.020, seg), SINK_H - 0.004),
        (o(ix - 0.002, iy - 0.002, 0.020, seg), SINK_FLOOR + 0.010),
        (o(ix - 0.016, iy - 0.016, 0.012, seg), SINK_FLOOR),
    ])
    drain = join(cylinder(0.024, 0.002, center=(0.0, 0.0, SINK_FLOOR + 0.001), segments=32))
    drain_hole = cylinder(0.017, 0.001, center=(0.0, 0.0, SINK_FLOOR + 0.0022), segments=32)
    water = prism(o(ix - 0.003, iy - 0.003, 0.020, seg), WATER_LEVEL - 0.0005, WATER_LEVEL)
    riser = (0.178, 0.0)
    tap_path = np.array([(riser[0], 0.0, 0.0), (riser[0], 0.0, 0.13), (riser[0], 0.0, 0.16), (riser[0] - 0.012, 0.0, 0.178),
                         (riser[0] - 0.04, 0.0, 0.178), (riser[0] - 0.058, 0.0, 0.16), (riser[0] - 0.06, 0.0, 0.142)])
    tap = join(tube(_catmull_rom(tap_path, 6), 0.011, 16), cylinder(0.024, 0.008, center=(riser[0], 0.0, 0.004)),
               cylinder(0.014, 0.022, center=(riser[0], 0.0, 0.12)),
               box((0.05, 0.010, 0.008), center=(riser[0] + 0.03, 0.0, 0.14), bevel=0.002))
    parts = [("basin", basin, "ceramic"), ("drain", drain, "chrome"), ("drain_hole", drain_hole, "dark"), ("water", water, "water"),
             ("tap", tap, "chrome")]
    w, h = SINK_WALL, SINK_H
    colliders = [
        collider("floor", (sx, sy, SINK_FLOOR), (0.0, 0.0, SINK_FLOOR / 2), "Sink"),
        collider("wall_back", (w, sy, h), (sx / 2 - w / 2, 0.0, h / 2), "Sink"),
        collider("wall_front", (w, sy, h), (-sx / 2 + w / 2, 0.0, h / 2), "Sink"),
        collider("wall_left", (sx, w, h), (0.0, sy / 2 - w / 2, h / 2), "Sink"),
        collider("wall_right", (sx, w, h), (0.0, -sy / 2 + w / 2, h / 2), "Sink"),
        collider("tap_riser", (0.026, 0.026, 0.18), (riser[0], 0.0, 0.09), "Sink"),
        collider("tap_spout", (0.07, 0.022, 0.03), (riser[0] - 0.035, 0.0, 0.17), "Sink"),
    ]
    path = write_usda(str(ASSETS / "sink.usda"), "Sink", parts, LOOKS, colliders, mass=1.0, friction=(0.5, 0.4), crease=50.0)
    _make_water_clear(path, "Sink", "water")
    text = Path(path).read_text()
    head = 'def Mesh "water" ('
    i = text.index("{", text.index(head))
    Path(path).write_text(text[:i + 1] + "\n            bool primvars:doNotCastShadows = 1" + text[i + 1:])
    return path


def write_wall() -> str:
    """Origin on the plate's front face, level with the table. A 2 mm plate on a tiled splashback; a slot for the pins."""
    wx, wy, wz = WALL_SIZE
    z = SOCKET[2]
    tiles = box((wx - 0.002, wy, wz), center=(0.002 + (wx - 0.002) / 2, 0.0, wz / 2))
    plate = box((0.002, 0.08, 0.08), center=(0.001, 0.0, z))
    ring = cylinder(0.0195, 0.0006, center=(-0.0003, 0.0, z), axis="x", segments=32)
    holes = join(*(cylinder(0.0032, 0.0006, center=(-0.0006, s * 0.0095, z), axis="x") for s in (-1, 1)))
    parts = [("tiles", tiles, "tile"), ("plate", plate, "plate"), ("ring", ring, "plate_ring"), ("holes", holes, "dark")]
    slot_y, slot_z, depth = 0.034, 0.008, 0.026  # loose sideways, so that a slightly crooked pull does not jam
    side = (wy - slot_y) / 2
    colliders = [
        collider("left", (depth, side, wz), (depth / 2, slot_y / 2 + side / 2, wz / 2), "Wall"),
        collider("right", (depth, side, wz), (depth / 2, -slot_y / 2 - side / 2, wz / 2), "Wall"),
        collider("above", (depth, slot_y, wz - z - slot_z / 2), (depth / 2, 0.0, (wz + z + slot_z / 2) / 2), "Wall"),
        collider("below", (depth, slot_y, z - slot_z / 2), (depth / 2, 0.0, (z - slot_z / 2) / 2), "Wall"),
        collider("back", (wx - depth, wy, wz), ((wx + depth) / 2, 0.0, wz / 2), "Wall"),
    ]
    return write_usda(str(ASSETS / "wall.usda"), "Wall", parts, LOOKS, colliders, mass=1.0, friction=(0.5, 0.4))


def write_towel() -> str:
    """Origin at the towel's middle, on the table. Three folds, each edge rounded, so its side reads as layers of cloth."""
    sx, sy, sz = TOWEL_SIZE
    layer = sz / 3
    folds = join(*(box((sx - 0.002 * k, sy - 0.002 * k, layer), center=(0.0, 0.0, (k + 0.5) * layer), bevel=0.004, round_xy=0.008)
                   for k in range(3)))
    return write_usda(str(ASSETS / "towel.usda"), "Towel", [("folds", folds, "terry")], LOOKS,
                      [collider("pad", TOWEL_SIZE, (0.0, 0.0, sz / 2), "Towel")], mass=1.0, friction=(0.9, 0.8), crease=60.0)


# -- the dryer, its cord and its plug ----------------------------------------------------------------------------------


def _dryer_visuals() -> list[tuple[str, Mesh, str]]:
    barrel = lathe([(0.0, -0.10), (0.031, -0.10), (0.0335, -0.097), (0.0335, -0.062), (0.032, -0.03), (0.032, 0.03), (0.030, 0.058),
                    (0.0262, 0.078), (0.0, 0.078)], 40).turned(-90.0, "x")
    nozzle = lathe([(0.0, 0.074), (0.0265, 0.074), (0.0225, 0.098), (0.0225, 0.102), (0.0, 0.102)], 40).turned(-90.0, "x")
    mouth = cylinder(0.019, 0.001, center=(0.0, 0.1025, 0.0), axis="y", segments=32)
    grille = cylinder(0.029, 0.001, center=(0.0, -0.1005, 0.0), axis="y", segments=40)
    ribs = join(*(cylinder(r, 0.0006, center=(0.0, -0.1012, 0.0), axis="y", segments=32) for r in (0.008,)))
    outline = uk._rounded_outline(HANDLE_Z, HANDLE_Y, 0.010, 6)
    handle = prism(outline, -HANDLE[1], -HANDLE[0], 0.003).turned(-90.0, "y")
    band = box((0.007, HANDLE_Y + 0.0012, HANDLE_Z + 0.0012), center=(HANDLE[0] + 0.008, 0.0, 0.0), bevel=0.0005, round_xy=0.0)
    switch = box((0.020, 0.012, 0.003), center=(-0.075, 0.0, HANDLE_Z / 2 + 0.001), bevel=0.0008)
    boot = lathe([(0.0, -0.0), (0.0095, 0.0), (0.0065, 0.016), (0.0045, 0.020), (0.0, 0.020)], 16).turned(90.0, "y").scaled((-1, 1, 1))
    boot = Mesh(boot.v + np.array([HANDLE[0], 0.0, 0.0]), boot.f[:, ::-1])
    return [("barrel", barrel, "body"), ("nozzle", nozzle, "body_dark"), ("mouth", mouth, "dark"), ("grille", grille, "body_dark"),
            ("hub", ribs, "dark"), ("handle", handle, "body"), ("band", band, "body_dark"), ("switch", switch, "switch"),
            ("boot", boot, "rubber")]


def _plug_visuals() -> list[tuple[str, Mesh, str]]:
    px, py, pz = PLUG_SIZE
    shell = box((px - 0.003, py, pz), center=(-0.0015, 0.0, 0.0), bevel=0.003, round_xy=0.008)
    collar = box((0.003, py, pz), center=(px / 2 - 0.0015, 0.0, 0.0), bevel=0.0008, round_xy=0.006)
    boot = cylinder(0.0065, 0.014, center=(-px / 2 - 0.004, 0.0, 0.0), axis="x", segments=16)
    pins = join(*(cylinder(0.002, 0.020, center=(px / 2 + 0.010, s * 0.0095, 0.0), axis="x", segments=12) for s in (-1, 1)))
    return [("shell", shell, "rubber"), ("collar", collar, "rubber"), ("boot", boot, "rubber"), ("pins", pins, "brass")]


def write_dryer() -> str:
    """The articulation, with its root prim at the barrel's middle in the dryer's own frame: x along the handle (its end at
    -x), y along the barrel (the nozzle at +y). Links are authored at their starting places in that frame."""
    root = "Dryer"
    r_body = rot_y(DRYER_TILT)
    origin = np.array([DRYER_XY[0], DRYER_XY[1], DRYER_Z])
    pts = (cord_world_points() - origin) @ r_body  # into the body's frame
    frames = _chain_frames(pts)
    n = len(frames)
    r_plug = r_body.T  # the plug is level in the world, pins along +x
    px = PLUG_SIZE[0]
    plug_mid = (np.array(PLUG_MID) - origin) @ r_body
    used = {"rubber", "brass", *(p[2] for p in _dryer_visuals())}
    looks = "".join(uk._material_usda(k, root, LOOKS[k]) for k in sorted(used))
    material = f"/{root}/PhysicsMaterial"

    def collide(kind: str, name: str, body: str) -> str:
        return f"""
        def {kind} "{name}" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
        )
        {{
            rel material:binding:physics = <{material}>
            uniform token purpose = "guide"
            float physxCollision:contactOffset = 0.004
            float physxCollision:restOffset = 0
{body}
        }}"""

    barrel_c = collide("Cylinder", "barrel_c", f"""            uniform token axis = "Y"
            double height = {BARREL_L:.6f}
            double radius = {BARREL_R:.6f}""")
    hx0, hx1 = HANDLE
    handle_c = collide("Cube", "handle_c", f"""            double size = 1
            float physxCollision:torsionalPatchRadius = 0.012
            float physxCollision:minTorsionalPatchRadius = 0.010
            double3 xformOp:translate = ({0.5 * (hx0 + hx1):.6f}, 0, 0)
            float3 xformOp:scale = ({hx1 - hx0:.6f}, {HANDLE_Y:.6f}, {HANDLE_Z:.6f})
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]""")
    dryer_meshes = "".join(uk._mesh_usda(*p, root, LOOKS[p[2]], 40.0, indent="        ") for p in _dryer_visuals())
    body = f"""
    def Xform "body" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsArticulationRootAPI", "PhysxArticulationAPI", "PhysxRigidBodyAPI"]
    )
    {{
        bool physxArticulation:enabledSelfCollisions = 1
        int physxArticulation:solverPositionIterationCount = 32
        int physxArticulation:solverVelocityIterationCount = 8
        float physxRigidBody:maxDepenetrationVelocity = 1
        float physics:mass = {DRYER_MASS:.4f}
        point3f physics:centerOfMass = {_fmt(DRYER_COM)}
{barrel_c}{handle_c}
{dryer_meshes}
    }}
"""
    links = []
    for i in range(n):
        mid = 0.5 * (pts[i] + pts[i + 1])
        length = float(np.linalg.norm(pts[i + 1] - pts[i]))
        q = _quat_wxyz(frames[i])
        cap = collide("Capsule", "capsule", f"""            uniform token axis = "X"
            double height = {max(length - 2 * CORD_R, 0.001):.6f}
            double radius = {CORD_R:.6f}""")
        links.append(f"""
    def Xform "link_{i:03d}" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
    )
    {{
        float physics:mass = {LINK_MASS:.5f}
        float physxRigidBody:maxDepenetrationVelocity = 1
        double3 xformOp:translate = {_fmt(mid)}
        quatd xformOp:orient = {_fmt(q)}
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
{cap}
        def Capsule "skin" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {{
            rel material:binding = </{root}/Looks/rubber>
            uniform token axis = "X"
            double height = {length:.6f}
            double radius = {CORD_R:.6f}
        }}
    }}
""")
    shell_c = collide("Cube", "shell_c", f"""            double size = 1
            float physxCollision:torsionalPatchRadius = 0.010
            float physxCollision:minTorsionalPatchRadius = 0.008
            float3 xformOp:scale = {_fmt(PLUG_SIZE)}
            uniform token[] xformOpOrder = ["xformOp:scale"]""")
    pin_c = collide("Cube", "pins_c", f"""            double size = 1
            double3 xformOp:translate = ({px / 2 + PIN_BLOCK[0] / 2:.6f}, 0, 0)
            float3 xformOp:scale = {_fmt(PIN_BLOCK)}
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]""")
    plug_meshes = "".join(uk._mesh_usda(*p, root, LOOKS[p[2]], 40.0, indent="        ") for p in _plug_visuals())
    links.append(f"""
    def Xform "plug" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
    )
    {{
        float physics:mass = {PLUG_MASS:.4f}
        float physxRigidBody:maxDepenetrationVelocity = 1
        double3 xformOp:translate = {_fmt(plug_mid)}
        quatd xformOp:orient = {_fmt(_quat_wxyz(r_plug))}
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
{shell_c}{pin_c}
{plug_meshes}
    }}
""")

    def joint(name, body0, body1, pos0, rot0, pos1, axis) -> str:
        return f"""
    def PhysicsRevoluteJoint "{name}" (
        prepend apiSchemas = ["PhysicsDriveAPI:angular"]
    )
    {{
        uniform token drive:angular:physics:type = "force"
        float drive:angular:physics:stiffness = 0
        float drive:angular:physics:damping = 0
        rel physics:body0 = </{root}/{body0}>
        rel physics:body1 = </{root}/{body1}>
        uniform token physics:axis = "{axis}"
        point3f physics:localPos0 = {_fmt(pos0)}
        quatf physics:localRot0 = {_fmt(rot0)}
        point3f physics:localPos1 = {_fmt(pos1)}
        quatf physics:localRot1 = (1, 0, 0, 0)
        float physics:lowerLimit = -150
        float physics:upperLimit = 150
    }}
"""

    joints = []
    for i in range(n):
        half = 0.5 * float(np.linalg.norm(pts[i + 1] - pts[i]))
        if i == 0:
            parent, r_p, o_p = "body", np.eye(3), np.zeros(3)
        else:
            parent, r_p, o_p = f"link_{i - 1:03d}", frames[i - 1], 0.5 * (pts[i - 1] + pts[i])
        joints.append(joint(f"cj_{i:03d}", parent, f"link_{i:03d}", r_p.T @ (pts[i] - o_p), _quat_wxyz(r_p.T @ frames[i]),
                            (-half, 0.0, 0.0), "Y" if i % 2 == 0 else "Z"))
    half = 0.5 * float(np.linalg.norm(pts[-1] - pts[-2]))
    joints.append(joint("cj_plug", f"link_{n - 1:03d}", "plug", (half, 0.0, 0.0), _quat_wxyz(frames[-1].T @ r_plug),
                        (-0.5 * px, 0.0, 0.0), "Y" if n % 2 == 0 else "Z"))
    text = f"""#usda 1.0
(
    defaultPrim = "{root}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}"
{{
    def Material "PhysicsMaterial" (
        prepend apiSchemas = ["PhysicsMaterialAPI"]
    )
    {{
        float physics:staticFriction = 0.9
        float physics:dynamicFriction = 0.8
        float physics:restitution = 0
    }}

    def Scope "Looks"
    {{{looks}
    }}
{body}{''.join(links)}{''.join(joints)}}}
"""
    os.makedirs(ASSETS, exist_ok=True)
    path = ASSETS / "dryer.usda"
    path.write_text(text)
    return str(path)


def cord_length() -> float:
    return float(np.linalg.norm(np.diff(cord_world_points(), axis=0), axis=1).sum())


def write_all() -> None:
    _make_textures()
    write_sink()
    write_wall()
    write_towel()
    write_dryer()
