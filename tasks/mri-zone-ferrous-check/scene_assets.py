"""The task's parts as plain-text USD files, written beside this file in assets/ (no pxr needed), with their textures in
textures/ (PIL).

- Three basins of one size class, each at its maker's catalogue size, with flared walls and a rolled rim: basin A, a
  stainless kidney dish 170 x 100 x 35 mm (Surtex HW-400-17) with a yellow MR Conditional label on its inside floor;
  basin B, a white rectangular enamelled-steel bowl with a blue rim, 159 x 124 x 38 mm (American Metalcraft EWB6);
  basin C, a stainless kidney basin 170 x 95 x 35 mm (Medicon 47.20.17). Which one the test magnet pulls is set in
  task.py, not here.
- The hand-held MRI test magnet: a red horseshoe magnet head, 1 x 1 x 1.5 in, on a black 8 in handle, 9.25 in overall,
  with a red MR Unsafe sticker on the handle (AliMed MRI Test Magnet Hand Held >=1000 Gauss).
- The ward cart's top (a white polypropylene tray), the scanner-room trolley's top (a white tray with a yellow MR
  Conditional sticker), a red MR Unsafe holding tote, and the doorway: red-and-white floor tape with an MRI Zone IV floor
  sign beyond it.

Colliders are boxes. A basin's thin sheet is given 3 mm wall boxes (its outside and inside surfaces stay within
1.5 mm of the real ones), so the solver has something to hold. Pure Python, NumPy and PIL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import usdkit as uk

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
TEXTURES = HERE / "textures"

# -- the basins ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Shape:
    """A basin's outline in its own frame: x across its width, y along its length, origin at the middle of its base."""

    name: str
    length: float            # rim length
    width: float             # rim width
    height: float            # depth
    mass: float              # kg
    power: float = 2.6       # the outline's superellipse power: 2.6 a kidney dish's round ends, 6 a rounded rectangle
    bend: float = 0.010      # the kidney's bend: the inner side is hollowed this much at the middle
    base_scale: tuple[float, float] = (0.80, 0.86)  # the floor's outline against the rim's, across and along

    def rim(self, n: int = 28) -> np.ndarray:
        return kidney(n, self.width, self.length, self.power, self.bend)

    def base(self, n: int = 28) -> np.ndarray:
        return kidney(n, self.width * self.base_scale[0], self.length * self.base_scale[1], self.power, self.bend)

    def width_at(self, y: float, z: float) -> float:
        """The outside width across x, at y along the basin and z above its base (for the grip)."""
        f = z / self.height
        out = (1 - f) * self.base(400) + f * self.rim(400)
        near = out[np.abs(out[:, 1] - y) < 0.004]
        return float(near[:, 0].max() - near[:, 0].min())


# masses are estimates: 0.16 kg for a 170 mm stainless dish (no maker publishes it), 0.14 kg for the enamelled bowl (its
# seller lists a shipping weight of 0.32 lb)
BASIN_A = Shape("basin_a", 0.170, 0.100, 0.035, 0.16)                       # Surtex HW-400-17
BASIN_B = Shape("basin_b", 0.159, 0.124, 0.038, 0.14, power=6.0, bend=0.0,  # American Metalcraft EWB6 (6 1/4 x 4 7/8 x
                base_scale=(0.84, 0.88))                                    # 1 1/2 in)
BASIN_C = Shape("basin_c", 0.170, 0.095, 0.035, 0.16)                       # Medicon 47.20.17
LABEL = 0.05             # basin A's MR Conditional label, square
WALL_T = 0.003           # collider wall thickness (the real sheet is under 1 mm)
FLOOR_T = 0.003
BEAD_H = 0.0045          # the rolled rim's height; its outside stands 1.2 mm beyond the rim line
N_OUT = 28               # outline points

# -- the test magnet ----------------------------------------------------------------------------------------------------

HEAD_W = 0.0381          # across the poles (1.5 in)
HEAD_H = 0.0254          # height (1 in)
HEAD_L = 0.0254          # along the handle (1 in)
HANDLE = 0.022           # square handle's side (not published; a typical hand-tool grip)
HANDLE_END = 0.235       # 9.25 in overall, from the pole face
CAP_L = 0.012            # a hanging cap at the handle's end, the head's height, so the magnet lies level
MAGNET_MASS = 0.30       # kg, an estimate: the maker does not publish it

# -- trays --------------------------------------------------------------------------------------------------------------

TRAY_T = 0.003
WARD_IN = (0.245, 0.415)       # ward cart top, inner x by y
WARD_LIP = 0.012
TROLLEY_IN = (0.180, 0.320)    # scanner-room trolley top, inner x by y
TROLLEY_DECAL = 0.045          # its MR Conditional sticker, centred at local y 3 x 0.045 (textures repeat every sticker size)
SIGN = 0.24                    # the Zone IV floor sign, square
TROLLEY_LIP = 0.025
TOTE_OUT = (0.12, 0.16, 0.08)  # red MR Unsafe holding tote, outside x, y, z


def superellipse(n: int, a: float, b: float, p: float = 2.6) -> np.ndarray:
    t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    return np.c_[a * np.sign(c) * np.abs(c) ** (2 / p), b * np.sign(s) * np.abs(s) ** (2 / p)]


def kidney(n: int, w: float, length: float, power: float = 2.6, bend: float = 0.010) -> np.ndarray:
    """A basin's outline in its own frame (x across, y along), counter-clockwise, its bounding box centred: a
    superellipse, hollowed by `bend` on one side for a kidney dish."""
    pts = superellipse(n, w / 2, length / 2, power)
    pts[:, 0] += bend * (1.0 - (2 * pts[:, 1] / length) ** 2)
    pts[:, 0] -= 0.5 * (pts[:, 0].max() + pts[:, 0].min())
    return pts


def _offset(outline: np.ndarray, d: float) -> np.ndarray:
    """The outline moved inward by d (counter-clockwise outline: inward is to the left of the direction of travel)."""
    nxt, prv = np.roll(outline, -1, axis=0), np.roll(outline, 1, axis=0)
    t = nxt - prv
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    inward = np.c_[-t[:, 1], t[:, 0]]
    return outline + d * inward


def _quat_from_axes(e1, e2, e3) -> tuple[float, float, float, float]:
    """(w, x, y, z) of the rotation whose columns are e1, e2, e3."""
    R = np.c_[e1, e2, e3]
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / n, x / n, y / n, z / n)


def turned_collider(name: str, size, center, quat_wxyz, root: str) -> str:
    """A box collider turned by a quaternion (not rendered: purpose "guide")."""
    sx, sy, sz = size
    cx, cy, cz = center
    w, x, y, z = quat_wxyz
    return f"""
    def Cube "{name}" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
    )
    {{
        rel material:binding:physics = </{root}/PhysicsMaterial>
        uniform token purpose = "guide"
        double size = 1
        float physxCollision:contactOffset = 0.003
        float physxCollision:restOffset = 0
        double3 xformOp:translate = ({cx:.6f}, {cy:.6f}, {cz:.6f})
        quatf xformOp:orient = ({w:.6f}, {x:.6f}, {y:.6f}, {z:.6f})
        float3 xformOp:scale = ({sx:.6f}, {sy:.6f}, {sz:.6f})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    }}"""


def basin_boxes(shape: Shape) -> list[tuple[tuple[float, float, float], tuple[float, float, float], np.ndarray]]:
    """A basin's collider boxes in its own frame (origin at the middle of its base): (size, centre, rotation)."""
    h = shape.height
    rim, base = shape.rim(N_OUT), shape.base(N_OUT)
    n = len(rim)
    boxes = []
    for i in range(n):
        j = (i + 1) % n
        r0, r1 = np.r_[rim[i], h], np.r_[rim[j], h]
        b0, b1 = np.r_[base[i], 0.0], np.r_[base[j], 0.0]
        e1 = (r1 - r0 + b1 - b0) / 2
        seg = float(np.linalg.norm(e1))
        e1 /= seg
        up = (r0 + r1 - b0 - b1) / 2
        up -= up.dot(e1) * e1
        height = float(np.linalg.norm(up))
        e3 = up / height
        e2 = np.cross(e3, e1)  # outward for a counter-clockwise outline
        centre = (r0 + r1 + b0 + b1) / 4 - e2 * (WALL_T / 2 - 0.0004)
        boxes.append(((seg + 0.004, WALL_T, height), tuple(centre), np.c_[e1, e2, e3]))
    # the rolled rim: a bead round the top whose outside is vertical, where a pinch or the magnet meets the basin first
    out = _offset(rim, -0.0012)
    for i in range(n):
        j = (i + 1) % n
        a, b = out[i], out[j]
        e1 = np.r_[b - a, 0.0]
        seg = float(np.linalg.norm(e1))
        e1 /= seg
        e3 = np.array([0.0, 0.0, 1.0])
        e2 = np.cross(e3, e1)
        centre = np.r_[(a + b) / 2, h - BEAD_H / 2] - e2 * (WALL_T / 2)
        boxes.append(((seg + 0.004, WALL_T, BEAD_H), tuple(centre), np.c_[e1, e2, e3]))
    # the floor: bands along y, each as wide as the base is at both its ends
    inner = _offset(base, 0.002)
    ys = np.linspace(inner[:, 1].min(), inner[:, 1].max(), 7)
    for y0, y1 in zip(ys[:-1], ys[1:]):
        band = inner[(inner[:, 1] >= y0 - 0.008) & (inner[:, 1] <= y1 + 0.008)]
        if len(band) < 2:
            continue
        left = max(_edge(inner, y, -1) for y in (y0, y1))
        right = min(_edge(inner, y, +1) for y in (y0, y1))
        if right - left < 0.01:
            continue
        boxes.append(((right - left, y1 - y0 + 0.002, FLOOR_T), ((left + right) / 2, (y0 + y1) / 2, FLOOR_T / 2), np.eye(3)))
    return boxes


def _edge(outline: np.ndarray, y: float, side: int) -> float:
    """The outline's x on its +x (side=1) or -x (side=-1) edge at y."""
    xs = []
    n = len(outline)
    for i in range(n):
        a, b = outline[i], outline[(i + 1) % n]
        if (a[1] - y) * (b[1] - y) <= 0 and a[1] != b[1]:
            t = (y - a[1]) / (b[1] - a[1])
            xs.append(a[0] + t * (b[0] - a[0]))
    if not xs:
        return 0.0
    return max(xs) if side > 0 else min(xs)


def basin_mesh(shape: Shape) -> uk.Mesh:
    """A thin shell: outside wall, rolled rim, inside wall, inside floor, outside bottom."""
    h = shape.height
    rim, base = shape.rim(64), shape.base(64)
    n = len(rim)
    bead = 0.0018
    rim_in = _offset(rim, 2 * bead)
    base_in = _offset(base, 0.0012)
    rings = [
        np.c_[base, np.zeros(n)],
        np.c_[_offset(rim, -0.0004), np.full(n, h - bead)],
        np.c_[_offset(rim, -0.0006), np.full(n, h)],
        np.c_[_offset(rim, bead), np.full(n, h + 0.0012)],
        np.c_[rim_in, np.full(n, h - bead)],
        np.c_[base_in, np.full(n, 0.0009)],
    ]
    v = np.vstack(rings + [np.array([[0.0, 0.0, 0.0009], [0.0, 0.0, 0.0]])])
    top, bottom = len(rings) * n, len(rings) * n + 1
    f = []
    for k in range(len(rings) - 1):
        for i in range(n):
            j = (i + 1) % n
            a, b, c, d = k * n + i, k * n + j, (k + 1) * n + j, (k + 1) * n + i
            f += [(a, b, c), (a, c, d)]
    last = (len(rings) - 1) * n
    for i in range(n):
        j = (i + 1) % n
        f += [(top, last + j, last + i), (bottom, i, j)]
    return uk.Mesh(v, np.array(f))


def rim_mesh(shape: Shape) -> uk.Mesh:
    """The rolled rim alone, a hair outside the shell, for a rim of another colour (the enamelled bowl's blue rim)."""
    rim = shape.rim(64)
    n = len(rim)
    h, bead = shape.height, 0.0018
    rings = [np.c_[_offset(rim, -0.0010), np.full(n, h - bead - 0.001)], np.c_[_offset(rim, -0.0012), np.full(n, h)],
             np.c_[_offset(rim, bead), np.full(n, h + 0.0016)], np.c_[_offset(rim, 2 * bead + 0.0005), np.full(n, h - bead - 0.001)]]
    v = np.vstack(rings)
    f = []
    for k in range(len(rings) - 1):
        for i in range(n):
            j = (i + 1) % n
            a, b, c, d = k * n + i, k * n + j, (k + 1) * n + j, (k + 1) * n + i
            f += [(a, b, c), (a, c, d)]
    return uk.Mesh(v, np.array(f))


# -- textures -----------------------------------------------------------------------------------------------------------


def _font(size: int):
    from PIL import ImageFont

    for name in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def _centered(draw, xy, text, font, fill) -> None:
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (x1 - x0) / 2 - x0, xy[1] - (y1 - y0) / 2 - y0), text, font=font, fill=fill)


def _save(im, name: str, centred: bool) -> None:
    """Save a picture drawn as seen from the robot (its up = world +x, its right = world -y). Textures map world x to the
    image's columns and world y to its rows (usdkit's planar projection), so it is turned a quarter turn clockwise; a
    sticker centred on a tile corner gets its picture rolled by half a tile."""
    import numpy as np
    from PIL import Image

    a = np.asarray(im.rotate(-90, expand=True))
    if centred:
        a = np.roll(a, (a.shape[0] // 2, a.shape[1] // 2), axis=(0, 1))
    Image.fromarray(a).save(TEXTURES / name)


def write_textures() -> None:
    from PIL import Image, ImageDraw

    TEXTURES.mkdir(exist_ok=True)
    # MR Unsafe (ASTM F2503): a red circle with a red bar across black "MR", on a white sticker
    im = Image.new("RGB", (256, 256), "white")
    d = ImageDraw.Draw(im)
    d.ellipse((20, 20, 236, 236), outline=(200, 20, 30), width=26)
    _centered(d, (128, 128), "MR", _font(96), (0, 0, 0))
    d.line((52, 204, 204, 52), fill=(200, 20, 30), width=26)
    _save(im, "mr_unsafe.png", True)
    # MR Conditional: a yellow triangle, black border, black "MR"
    im = Image.new("RGB", (256, 256), "white")
    d = ImageDraw.Draw(im)
    d.polygon([(128, 14), (244, 230), (12, 230)], fill=(0, 0, 0))
    d.polygon([(128, 44), (220, 214), (36, 214)], fill=(250, 210, 0))
    _centered(d, (128, 160), "MR", _font(74), (0, 0, 0))
    _save(im, "mr_conditional.png", True)
    # basin A's label: the same triangle over the condition it was labelled for
    im = Image.new("RGB", (256, 256), "white")
    d = ImageDraw.Draw(im)
    d.polygon([(128, 6), (232, 190), (24, 190)], fill=(0, 0, 0))
    d.polygon([(128, 32), (210, 176), (46, 176)], fill=(250, 210, 0))
    _centered(d, (128, 132), "MR", _font(62), (0, 0, 0))
    _centered(d, (128, 222), "1.5 T", _font(52), (0, 0, 0))
    _save(im, "mr_conditional_label.png", True)
    # the MRI floor sign beyond the doorway (a 768 x 384 panel on a square sheet)
    im = Image.new("RGB", (768, 768), (250, 210, 0))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 767, 191), fill=(250, 210, 0))
    o = 192
    d.rectangle((14, 14 + o, 754, 370 + o), outline=(0, 0, 0), width=14)
    d.rectangle((14, 14 + o, 754, 120 + o), fill=(0, 0, 0))
    _centered(d, (384, 67 + o), "DANGER  MRI  ZONE IV", _font(50), (250, 210, 0))
    _centered(d, (384, 180 + o), "1.5 T MAGNET ALWAYS ON", _font(46), (0, 0, 0))
    _centered(d, (384, 252 + o), "NO UNTESTED OR UNLABELLED", _font(40), (0, 0, 0))
    _centered(d, (384, 318 + o), "METAL PAST THIS LINE", _font(40), (0, 0, 0))
    _save(im, "zone_sign.png", False)
    # red-and-white floor tape: diagonal stripes that tile
    im = Image.new("RGB", (128, 128), "white")
    d = ImageDraw.Draw(im)
    for k in range(-2, 4):
        x = k * 64
        d.polygon([(x, 128), (x + 32, 128), (x + 160, 0), (x + 128, 0)], fill=(205, 25, 30))
    _save(im, "tape.png", False)


def _decal(size: float, z: float) -> uk.Mesh:
    """A flat square sticker centred on the origin, lying at height z. Move it only by whole multiples of its size, so
    its (rolled) texture stays centred on it."""
    return uk.box((size, size, 0.0006), center=(0.0, 0.0, z))


# -- files --------------------------------------------------------------------------------------------------------------


def write_basin(shape: Shape) -> str:
    """Basin A: satin stainless with its label; basin B: white enamel with a blue rim; basin C: satin stainless."""
    name = shape.name
    if shape is BASIN_B:
        looks = {"body": uk.Look(color=(0.93, 0.93, 0.90), roughness=0.18), "rim": uk.Look(color=(0.05, 0.12, 0.45), roughness=0.2)}
        parts = [("shell", basin_mesh(shape), "body"), ("rim", rim_mesh(shape), "rim")]
    else:
        looks = {"body": uk.Look(color=(0.78, 0.78, 0.80), metallic=1.0, roughness=0.34)}
        parts = [("shell", basin_mesh(shape), "body")]
    if shape is BASIN_A:
        looks["label"] = uk.Look(texture="mr_conditional_label.png", color=(1.0, 1.0, 1.0), tile=LABEL)
        parts.append(("label", _decal(LABEL, 0.0009 + 0.0004), "label"))
    cols = []
    for k, (size, centre, R) in enumerate(basin_boxes(shape)):
        cols.append(turned_collider(f"c{k}", size, centre, _quat_from_axes(R[:, 0], R[:, 1], R[:, 2]), name))
    return uk.write_usda(str(ASSETS / f"{name}.usda"), name, parts, looks, cols, mass=shape.mass, friction=(0.3, 0.25),
                         solver_iters=32)


def magnet_parts() -> tuple[list, list]:
    """Visual parts and collider boxes of the magnet in its own frame: origin at the middle of the pole face, the handle
    along +y, the pole face looking along -y."""
    red = "red"
    leg = 0.012
    parts = [
        ("leg_a", uk.box((leg, HEAD_L - 0.003, HEAD_H), center=(-(HEAD_W - leg) / 2, HEAD_L / 2 + 0.0015, 0.0), bevel=0.0015), red),
        ("leg_b", uk.box((leg, HEAD_L - 0.003, HEAD_H), center=((HEAD_W - leg) / 2, HEAD_L / 2 + 0.0015, 0.0), bevel=0.0015), red),
        ("bridge", uk.box((HEAD_W, 0.010, HEAD_H), center=(0.0, HEAD_L - 0.005, 0.0), bevel=0.0015), red),
        ("pole_a", uk.box((leg, 0.003, HEAD_H), center=(-(HEAD_W - leg) / 2, 0.0015, 0.0)), "pole"),
        ("pole_b", uk.box((leg, 0.003, HEAD_H), center=((HEAD_W - leg) / 2, 0.0015, 0.0)), "pole"),
        ("ferrule", uk.box((0.026, 0.008, 0.026), center=(0.0, HEAD_L + 0.004, 0.0), bevel=0.001), "pole"),
        ("handle", uk.box((HANDLE, HANDLE_END - HEAD_L - 0.008 - CAP_L, HANDLE),
                          center=(0.0, (HEAD_L + 0.008 + HANDLE_END - CAP_L) / 2, 0.0), bevel=0.002, round_xy=0.004), "black"),
        ("cap", uk.box((HANDLE, CAP_L, HEAD_H), center=(0.0, HANDLE_END - CAP_L / 2, 0.0), bevel=0.002, round_xy=0.004), "black"),
        ("sticker", _decal(0.020, HANDLE / 2 + 0.0003).moved((0.0, 0.060, 0.0)), "unsafe"),
    ]
    cols = [
        ((HEAD_W, HEAD_L + 0.008, HEAD_H), (0.0, (HEAD_L + 0.008) / 2, 0.0)),
        ((HANDLE, HANDLE_END - HEAD_L - 0.008 - CAP_L, HANDLE), (0.0, (HEAD_L + 0.008 + HANDLE_END - CAP_L) / 2, 0.0)),
        ((HANDLE, CAP_L, HEAD_H), (0.0, HANDLE_END - CAP_L / 2, 0.0)),
    ]
    return parts, cols


def write_magnet() -> str:
    looks = {"red": uk.Look(color=(0.62, 0.04, 0.04), roughness=0.45, metallic=0.2),
             "pole": uk.Look(color=(0.55, 0.55, 0.57), metallic=1.0, roughness=0.35),
             "black": uk.Look(color=(0.03, 0.03, 0.03), roughness=0.55),
             "unsafe": uk.Look(texture="mr_unsafe.png", color=(1.0, 1.0, 1.0), tile=0.020)}
    parts, boxes = magnet_parts()
    cols = [uk.collider(f"c{k}", s, c, "magnet") for k, (s, c) in enumerate(boxes)]
    return uk.write_usda(str(ASSETS / "magnet.usda"), "magnet", parts, looks, cols, mass=MAGNET_MASS, friction=(0.3, 0.25),
                         solver_iters=32)


def tray_boxes(inner, lip: float) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """A tray's floor and four walls, in its own frame (origin at the middle of its floor's underside)."""
    ix, iy = inner
    t = TRAY_T
    h = lip + t
    return [
        ((ix + 2 * t, iy + 2 * t, t), (0.0, 0.0, t / 2)),
        ((t, iy + 2 * t, h), (-(ix + t) / 2, 0.0, h / 2)),
        ((t, iy + 2 * t, h), ((ix + t) / 2, 0.0, h / 2)),
        ((ix, t, h), (0.0, -(iy + t) / 2, h / 2)),
        ((ix, t, h), (0.0, (iy + t) / 2, h / 2)),
    ]


def write_tray(name: str, inner, lip: float, look: uk.Look, decals=()) -> str:
    looks = {"pp": look}
    parts = [(f"p{k}", uk.box(s, center=c, bevel=0.0008), "pp") for k, (s, c) in enumerate(tray_boxes(inner, lip))]
    for k, (tex, size, xy) in enumerate(decals):
        looks[f"d{k}"] = uk.Look(texture=tex, color=(1.0, 1.0, 1.0), tile=size)
        parts.append((f"decal{k}", _decal(size, TRAY_T + 0.0004).moved((xy[0], xy[1], 0.0)), f"d{k}"))
    cols = [uk.collider(f"c{k}", s, c, name) for k, (s, c) in enumerate(tray_boxes(inner, lip))]
    # a rigid body like the others; task.py spawns it kinematic (mass None in its UsdBody), so it never moves
    return uk.write_usda(str(ASSETS / f"{name}.usda"), name, parts, looks, cols, mass=1.0, friction=(0.3, 0.25))


def write_tote() -> str:
    sx, sy, sz = TOTE_OUT
    t = 0.004
    boxes = [((sx, sy, t), (0.0, 0.0, t / 2)), ((t, sy, sz), (-(sx - t) / 2, 0.0, sz / 2)), ((t, sy, sz), ((sx - t) / 2, 0.0, sz / 2)),
             ((sx, t, sz), (0.0, -(sy - t) / 2, sz / 2)), ((sx, t, sz), (0.0, (sy - t) / 2, sz / 2))]
    looks = {"red": uk.Look(color=(0.55, 0.03, 0.03), roughness=0.5),
             "unsafe": uk.Look(texture="mr_unsafe.png", color=(1.0, 1.0, 1.0), tile=0.06)}
    parts = [(f"p{k}", uk.box(s, center=c, bevel=0.001), "red") for k, (s, c) in enumerate(boxes)]
    parts.append(("label", _decal(0.06, t + 0.0004), "unsafe"))
    cols = [uk.collider(f"c{k}", s, c, "tote") for k, (s, c) in enumerate(boxes)]
    return uk.write_usda(str(ASSETS / "tote.usda"), "tote", parts, looks, cols, mass=1.0)


def write_doorway(length_y: float = 1.20, tape_w: float = 0.05) -> str:
    """The doorway's floor tape (along y, 5 cm wide, from local x 0 to 0.05), with a collider as thin as the tape."""
    looks = {"tape": uk.Look(texture="tape.png", color=(1.0, 1.0, 1.0), tile=tape_w)}
    tape = uk.box((tape_w, length_y, 0.0006), center=(tape_w / 2, 0.0, 0.0003))
    cols = [uk.collider("c0", (tape_w, length_y, 0.0006), (tape_w / 2, 0.0, 0.0003), "doorway")]
    return uk.write_usda(str(ASSETS / "doorway.usda"), "doorway", [("tape", tape, "tape")], looks, cols, mass=0.1)


def write_sign() -> str:
    """The MRI Zone IV floor sign, SIGN square, from local (0, 0) to (SIGN, SIGN), lying flat."""
    looks = {"sign": uk.Look(texture="zone_sign.png", color=(1.0, 1.0, 1.0), tile=SIGN)}
    sign = uk.box((SIGN, SIGN, 0.0008), center=(SIGN / 2, SIGN / 2, 0.0004))
    cols = [uk.collider("c0", (SIGN, SIGN, 0.0008), (SIGN / 2, SIGN / 2, 0.0004), "sign")]
    return uk.write_usda(str(ASSETS / "sign.usda"), "sign", [("sign", sign, "sign")], looks, cols, mass=0.1)


def write_all() -> None:
    """Write the textures and USD files, touching only those whose content changes (several runs may import this file
    at once, and a renderer must never read a file while it is being rewritten)."""
    global ASSETS, TEXTURES
    import tempfile

    ASSETS.mkdir(exist_ok=True)
    TEXTURES.mkdir(exist_ok=True)
    real_assets, real_textures = ASSETS, TEXTURES
    with tempfile.TemporaryDirectory() as tmp:
        ASSETS, TEXTURES = Path(tmp) / "assets", Path(tmp) / "textures"
        ASSETS.mkdir()
        TEXTURES.mkdir()
        try:
            _write_all()
        finally:
            ASSETS, TEXTURES = real_assets, real_textures
        for sub, real in (("textures", real_textures), ("assets", real_assets)):
            for f in sorted((Path(tmp) / sub).iterdir()):
                data = f.read_bytes()
                if sub == "assets":   # the USD files name their textures relative to themselves (no machine paths)
                    data = data.replace((str(real_textures) + "/").encode(), b"../textures/")
                    data = data.replace((str(Path(tmp) / "textures") + "/").encode(), b"../textures/")
                target = real / f.name
                if not target.exists() or target.read_bytes() != data:
                    target.write_bytes(data)


def _write_all() -> None:
    write_textures()
    for shape in (BASIN_A, BASIN_B, BASIN_C):
        write_basin(shape)
    write_magnet()
    write_tray("ward", WARD_IN, WARD_LIP, uk.Look(color=(0.86, 0.86, 0.84), roughness=0.55))
    write_tray("trolley", TROLLEY_IN, TROLLEY_LIP, uk.Look(color=(0.90, 0.90, 0.88), roughness=0.55),
               decals=(("mr_conditional.png", TROLLEY_DECAL, (0.0, 3 * TROLLEY_DECAL)),))
    write_tote()
    write_doorway()
    write_sign()


if __name__ == "__main__":
    write_all()
    for shape in (BASIN_A, BASIN_B, BASIN_C):
        print(shape.name, "boxes", len(basin_boxes(shape)), "rim width 2.4 cm from the end",
              round(shape.width_at(shape.length / 2 - 0.024, shape.height) * 1000, 1), "mm")
