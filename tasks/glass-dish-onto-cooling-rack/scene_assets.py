"""The glass baking dish, the hot plate, the wire cooling rack and the washing-up bowl: meshes, materials and colliders as
plain-text USD.

Frames (metres, z up), each origin on the surface the object stands on, under its middle:
- dish: its two moulded end handles along y.
- hot plate, rack, bowl: their long side as task.py places them.

Sizes are the makers' published sizes where given; the handle's thickness, the wall thickness and the dish's mass are
engineering estimates (named in the README). Pure Python and NumPy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from usdkit import Look, Mesh, box, collider, cylinder, join, prism, write_usda

HERE = Path(__file__).resolve().parent

# -- glass baking dish: Pyrex Easy Grab 8 in square (2 qt) ----------------------------------------------------------------
DISH_TOP = 0.203        # outer size of the body at the top, square (8.0 in)
DISH_BOTTOM = 0.170     # outer size at the floor: the walls slope in
DISH_H = 0.070          # floor to rim top (2.75 in)
DISH_WALL = 0.005       # estimate
DISH_FLOOR = 0.005      # estimate
DISH_CORNER = 0.025     # corner radius at the top
FLANGE = 0.0115         # the rim flange stands this far out of the wall on the two plain sides (8.9 in across)
FLANGE_T = 0.006
HANDLE_OUT = 0.038      # each moulded handle stands this far out of the wall (11.0 in across both handles)
HANDLE_W = 0.150        # handle width along x (estimate from photographs)
HANDLE_T = 0.008        # handle thickness (estimate)
HANDLE_Z = DISH_H - HANDLE_T / 2   # the handle's mid-plane: its top level with the rim
DISH_MASS = 1.30        # estimate (retailer's shipping weight 3.1 lb includes the box)
DISH_ACROSS = DISH_TOP + 2 * HANDLE_OUT   # 0.279 m across the handles

# -- hot plate: Cuisinart CB-30 single cast-iron burner -------------------------------------------------------------------
PLATE_SIZE = (0.28, 0.29, 0.06)   # along x, y and its height to the top of the cast-iron plate
PLATE_DISC_R = 0.095

# -- cork trivet on the hot plate: IKEA HEAT, 19 cm round (thickness an estimate) ----------------------------------------
TRIVET_R = 0.095
TRIVET_T = 0.010

# -- oven mitt on the counter: a quilted cotton oven mitt, about 30 x 17 cm, lying flat (visual cue only) ----------------
MITT_SIZE = (0.30, 0.17, 0.025)

# -- cooling rack: Last Confection 10 x 15 in stainless wire rack --------------------------------------------------------
RACK_SIZE = (0.381, 0.254)        # along x and y: its long side along x
RACK_H = 0.025                    # to the top of the wires
WIRE_R = 0.0015
RACK_CROSS = 16                   # wires running along x, spaced along y
RACK_RAILS = 3                    # wires running along y underneath

# -- washing-up bowl: Rubbermaid 2951-AR 11.4 qt dishpan -----------------------------------------------------------------
BOWL_OUT = (0.319, 0.367)         # along x and y at the rim (its long side along y)
BOWL_BASE = (0.27, 0.32)
BOWL_H = 0.144
BOWL_WALL = 0.004
WATER_H = 0.08

LOOKS = {
    "glass": Look(color=(0.80, 0.90, 0.92), roughness=0.05, opacity=0.45),
    "glass_edge": Look(color=(0.55, 0.75, 0.78), roughness=0.08, opacity=0.7),
    "iron": Look(color=(0.05, 0.05, 0.05), roughness=0.6, metallic=0.4),
    "steel": Look(color=(0.75, 0.76, 0.78), roughness=0.3, metallic=0.9),
    "knob": Look(color=(0.02, 0.02, 0.02), roughness=0.4),
    "pan": Look(color=(0.92, 0.92, 0.90), roughness=0.45),
    "cork": Look(color=(0.62, 0.45, 0.28), roughness=0.9),
    "mitt": Look(color=(0.70, 0.08, 0.06), roughness=0.85),
    "mitt_cuff": Look(color=(0.92, 0.90, 0.86), roughness=0.85),
    "water": Look(color=(0.45, 0.62, 0.75), roughness=0.02, opacity=0.55),
}


def _square(s: float, r: float, seg: int = 4) -> np.ndarray:
    """A rounded square outline, counter-clockwise."""
    h = s / 2 - r
    pts = []
    for cx, cy, a0 in ((h, h, 0.0), (-h, h, 90.0), (-h, -h, 180.0), (h, -h, 270.0)):
        for k in range(seg + 1):
            a = np.radians(a0 + 90.0 * k / seg)
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return np.array(pts)


def _shell(o0: np.ndarray, o1: np.ndarray, z0: float, z1: float) -> Mesh:
    """An open surface between two outlines of the same length (bottom at z0, top at z1)."""
    n = len(o0)
    v = np.vstack([np.c_[o0, np.full(n, z0)], np.c_[o1, np.full(n, z1)]])
    f = []
    for i in range(n):
        j = (i + 1) % n
        f += [(i, j, n + j), (i, n + j, n + i)]
    return Mesh(v, np.array(f))


def _flip(m: Mesh) -> Mesh:
    return Mesh(m.v, m.f[:, ::-1])


def dish_parts() -> list:
    r0, r1 = DISH_CORNER * DISH_BOTTOM / DISH_TOP, DISH_CORNER
    out_b, out_t = _square(DISH_BOTTOM, r0), _square(DISH_TOP, r1)
    in_b = _square(DISH_BOTTOM - 2 * DISH_WALL, max(r0 - DISH_WALL, 0.004))
    in_t = _square(DISH_TOP - 2 * DISH_WALL, r1 - DISH_WALL)
    outer = _shell(out_b, out_t, 0.0, DISH_H)
    inner = _flip(_shell(in_b, in_t, DISH_FLOOR, DISH_H))
    floor = prism(out_b, 0.0, DISH_FLOOR)
    lip = _shell(out_t, in_t, DISH_H, DISH_H)
    body = join(outer, inner, floor, lip, _flip(lip))
    flanges = [box((FLANGE, DISH_TOP - 0.02, FLANGE_T), (sx * (DISH_TOP / 2 + FLANGE / 2 - 0.002), 0.0, DISH_H - FLANGE_T / 2),
                   bevel=0.002) for sx in (-1.0, 1.0)]
    handles = [box((HANDLE_W, HANDLE_OUT + 0.004, HANDLE_T), (0.0, sy * (DISH_TOP / 2 + HANDLE_OUT / 2 - 0.002), HANDLE_Z),
                   bevel=0.003, round_xy=0.012) for sy in (-1.0, 1.0)]
    return [("body", body, "glass"), ("rim", join(*flanges), "glass_edge"), ("handles", join(*handles), "glass_edge")]


def dish_colliders(root: str) -> list[str]:
    """The floor, four sloping walls, the two flanges and the two handles, each a box (the walls tilted ~13 deg are
    approximated by boxes at their mean position: within 2 mm of the glass where a hand or the rack touches it)."""
    cols = [collider("c_floor", (DISH_BOTTOM - 0.01, DISH_BOTTOM - 0.01, DISH_FLOOR), (0, 0, DISH_FLOOR / 2), root)]
    mid = (DISH_TOP + DISH_BOTTOM) / 4 - DISH_WALL / 2
    span = (DISH_TOP + DISH_BOTTOM) / 2 - 0.02
    for k, (sx, sy) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
        size = (DISH_WALL, span, DISH_H - DISH_FLOOR) if sx else (span, DISH_WALL, DISH_H - DISH_FLOOR)
        cols.append(collider(f"c_wall{k}", size, (sx * mid, sy * mid, DISH_FLOOR + (DISH_H - DISH_FLOOR) / 2), root))
    for k, sx in enumerate((-1.0, 1.0)):
        cols.append(collider(f"c_flange{k}", (FLANGE, DISH_TOP - 0.02, FLANGE_T),
                             (sx * (DISH_TOP / 2 + FLANGE / 2 - 0.002), 0.0, DISH_H - FLANGE_T / 2), root))
    for k, sy in enumerate((-1.0, 1.0)):
        cols.append(collider(f"c_handle{k}", (HANDLE_W - 0.01, HANDLE_OUT + 0.004, HANDLE_T),
                             (0.0, sy * (DISH_TOP / 2 + HANDLE_OUT / 2 - 0.002), HANDLE_Z), root))
    return cols


def plate_parts() -> list:
    sx, sy, h = PLATE_SIZE
    housing = box((sx, sy, h - 0.012), (0, 0, (h - 0.012) / 2), bevel=0.004, round_xy=0.02)
    disc = cylinder(PLATE_DISC_R, 0.012, (0.0, 0.0, h - 0.006), segments=40, bevel=0.002)
    rings = [cylinder(r, 0.0008, (0.0, 0.0, h + 0.0004), segments=40) for r in (0.03, 0.06)]
    knob = cylinder(0.014, 0.018, (-sx / 2 - 0.004, 0.08, 0.025), axis="x", segments=20, bevel=0.002)
    trivet = cylinder(TRIVET_R, TRIVET_T, (0.0, 0.0, h + TRIVET_T / 2 + 0.0008), segments=48, bevel=0.0015)
    return [("housing", housing, "steel"), ("disc", disc, "iron"), ("rings", join(*rings), "knob"), ("knob", knob, "knob"),
            ("trivet", trivet, "cork")]


def plate_colliders(root: str) -> list[str]:
    """The hot plate as a box, and the cork trivet on it as a box inside its round outline (the dish's 17 cm base rests on it)."""
    sx, sy, h = PLATE_SIZE
    t = 2 * TRIVET_R / np.sqrt(2.0)
    return [collider("c_plate", (sx, sy, h), (0, 0, h / 2), root),
            collider("c_trivet", (t, t, TRIVET_T), (0, 0, h + 0.0008 + TRIVET_T / 2), root)]


def mitt_parts() -> list:
    """A flat oven mitt: a rounded hand with a thumb, a white cuff. Origin on the counter under its middle; fingers along +x."""
    lx, ly, h = MITT_SIZE
    hand = box((lx * 0.72, ly * 0.78, h), (lx * 0.08, -ly * 0.06, h / 2), bevel=0.008, round_xy=0.05)
    thumb = box((lx * 0.34, ly * 0.30, h * 0.9), (lx * 0.05, ly * 0.34, h * 0.45), bevel=0.006, round_xy=0.024)
    cuff = box((lx * 0.24, ly * 0.82, h * 0.8), (-lx * 0.38, -ly * 0.04, h * 0.4), bevel=0.004, round_xy=0.01)
    return [("hand", join(hand, thumb), "mitt"), ("cuff", cuff, "mitt_cuff")]


def mitt_colliders(root: str) -> list[str]:
    lx, ly, h = MITT_SIZE
    return [collider("c_mitt", (lx * 0.9, ly * 0.8, h * 0.6), (0, 0, h * 0.3), root)]


def _rack_wires() -> list[tuple[tuple, tuple]]:
    """(size, centre) of each wire as a box: a frame, wires along x on top, rails along y under them, four feet."""
    lx, ly = RACK_SIZE
    d = 2 * WIRE_R
    top = RACK_H - WIRE_R
    out = []
    for k in range(RACK_CROSS):
        y = -ly / 2 + 0.01 + k * (ly - 0.02) / (RACK_CROSS - 1)
        out.append(((lx - 0.004, d, d), (0.0, y, top)))
    for sy in (-1.0, 1.0):
        out.append(((lx, 2 * d, 2 * d), (0.0, sy * (ly / 2 - 0.002), top - d / 2)))
    for k in range(RACK_RAILS):
        x = -lx / 2 + 0.006 + k * (lx - 0.012) / (RACK_RAILS - 1)
        out.append(((2 * d, ly, 2 * d), (x, 0.0, top - 1.5 * d)))
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            out.append(((2 * d, 2 * d, RACK_H - 2 * d), (sx * (lx / 2 - 0.006), sy * (ly / 2 - 0.02), (RACK_H - 2 * d) / 2)))
    return out


def rack_parts() -> list:
    return [("wires", join(*[box(s, c) for s, c in _rack_wires()]), "steel")]


def rack_colliders(root: str) -> list[str]:
    return [collider(f"c_w{i:02d}", s, c, root) for i, (s, c) in enumerate(_rack_wires())]


def _bowl_walls() -> list[tuple[tuple, tuple]]:
    ox, oy = BOWL_OUT
    w, h = BOWL_WALL, BOWL_H
    return [((BOWL_BASE[0], BOWL_BASE[1], w), (0, 0, w / 2)),
            ((ox, w, h), (0, oy / 2 - w / 2, h / 2)), ((ox, w, h), (0, -oy / 2 + w / 2, h / 2)),
            ((w, oy - 2 * w, h), (ox / 2 - w / 2, 0, h / 2)), ((w, oy - 2 * w, h), (-ox / 2 + w / 2, 0, h / 2))]


def bowl_parts() -> list:
    ob, ot = _square(BOWL_BASE[0], 0.03), _square(BOWL_OUT[0], 0.035)
    sy_b, sy_t = BOWL_BASE[1] / BOWL_BASE[0], BOWL_OUT[1] / BOWL_OUT[0]
    ob, ot = ob * np.array([1.0, sy_b]), ot * np.array([1.0, sy_t])
    ib, it = ob * 0.97, ot * (1 - 2 * BOWL_WALL / BOWL_OUT[0])
    outer = _shell(ob, ot, 0.0, BOWL_H)
    inner = _flip(_shell(ib, it, BOWL_WALL, BOWL_H))
    rim = join(_shell(ot, it, BOWL_H, BOWL_H), _flip(_shell(ot, it, BOWL_H, BOWL_H)))
    base = prism(ob, 0.0, BOWL_WALL)
    s = WATER_H / BOWL_H
    wo = ib + s * (it - ib)
    water = prism(np.vstack([ib * 0.995]), BOWL_WALL, BOWL_WALL + 0.0005)  # thin film on the floor
    water = join(_shell(ib * 0.995, wo * 0.995, BOWL_WALL, WATER_H), prism(wo * 0.995, WATER_H - 0.0005, WATER_H), water)
    return [("pan", join(outer, inner, rim, base), "pan"), ("water", water, "water")]


def bowl_colliders(root: str) -> list[str]:
    ix, iy = BOWL_OUT[0] - 0.03, BOWL_OUT[1] - 0.03
    cols = [collider(f"c_b{i}", s, c, root) for i, (s, c) in enumerate(_bowl_walls())]
    cols.append(collider("c_water", (ix, iy, WATER_H - BOWL_WALL), (0, 0, BOWL_WALL + (WATER_H - BOWL_WALL) / 2), root))
    return cols


GLASS_FRICTION = (0.5, 0.45)   # soda-lime glass on the gripper's pads: PhysX averages with the pad material
STEEL_FRICTION = (0.4, 0.35)


def write_all() -> None:
    a = HERE / "assets"
    write_usda(str(a / "dish.usda"), "dish", dish_parts(), LOOKS, dish_colliders("dish"), mass=DISH_MASS,
               friction=GLASS_FRICTION, solver_iters=32)
    # fixed parts are written as rigid bodies (the harness puts a contact sensor on every object); task.py makes them kinematic
    write_usda(str(a / "plate.usda"), "plate", plate_parts(), LOOKS, plate_colliders("plate"), mass=1.0, friction=(0.5, 0.45))
    write_usda(str(a / "mitt.usda"), "mitt", mitt_parts(), LOOKS, mitt_colliders("mitt"), mass=1.0, friction=(0.6, 0.5))
    write_usda(str(a / "rack.usda"), "rack", rack_parts(), LOOKS, rack_colliders("rack"), mass=1.0, friction=STEEL_FRICTION)
    write_usda(str(a / "bowl.usda"), "bowl", bowl_parts(), LOOKS, bowl_colliders("bowl"), mass=1.0, friction=(0.5, 0.45))
