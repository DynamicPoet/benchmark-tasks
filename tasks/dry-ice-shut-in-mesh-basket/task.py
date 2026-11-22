"""A child's drink made to fog with dry ice, the pellet shut inside a lidded mesh brew basket hung in the glass, never loose in it.

On the station table: a small stainless dish holding a dry-ice pellet, a lidded stainless mesh brew basket with two flat
handles, and the child's place: a place mat with a cork coaster and a rocks glass of lemonade on it. The pellet goes into the
basket, the basket hangs in the glass by its handles, and the lid sits in the basket's mouth.

    python demokit.py task.py OUT --mode full        the scripted demo
    python demokit.py task.py OUT --mode failure     the pellet dropped straight into the glass (scores 0)
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import demokit as d  # noqa: E402
import steptime  # noqa: E402
import mechkit as m  # noqa: E402

SCENE = "station"
INSTRUCTION = "Make the child's drink fog with a piece of the dry ice."
SUCCESS_HOLD_S = 0.0  # the scorer holds the end state itself
SCORE_OVER = "end"
TOP = d.TABLE_TOP
TRACE = bool(os.environ.get("DRY_TRACE"))

LOOKS = {
    "mesh": m.Look(m.rgb(0.80, 0.81, 0.83), metallic=0.9, roughness=0.35, opacity=0.5),
    "steel": m.Look(m.rgb(0.70, 0.71, 0.73), metallic=0.9, roughness=0.3),
    "lidc": m.Look(m.rgb(0.20, 0.45, 0.75), roughness=0.4),
    "ice": m.Look(m.rgb(0.96, 0.97, 0.98), roughness=0.8),
    "glass": m.Look(glass=(0.95, 0.97, 0.97)),
    "lemonade": m.Look(m.rgb(0.98, 0.86, 0.35), roughness=0.2, opacity=0.4),
    "cork": m.Look(m.rgb(0.70, 0.52, 0.33), roughness=0.9),
    "mat": m.Look(m.rgb(0.98, 0.55, 0.70), roughness=0.5),
    "matdots": m.Look(m.rgb(0.35, 0.75, 0.95), roughness=0.5),
    "card": m.Look(m.rgb(0.97, 0.97, 0.95), roughness=0.9),
}

# -- lidded mesh brew basket: part frame at the basket's floor centre, the lid a separate cap over its rim -------------------------
B_R, B_H, B_WALL = 0.0318, 0.076, 0.002     # basket 2.5 in across, 3 in tall (listing); mesh wall drawn 2 mm (real about 0.5 mm, estimate)
B_FLANGE = 0.0340                           # rolled rim
HANDLE_SPAN, HANDLE_W, HANDLE_T = 0.127, 0.012, 0.003   # two flat handles, 5 in tip to tip (listing), resting on a cup's rim
L_R, L_H, L_T = 0.038, 0.019, 0.003         # coaster lid 3 in across, 0.75 in tall (listing): a 3 mm disc on a plug that fits inside the mouth (estimate)
PLUG_RO, PLUG_RI = B_R - B_WALL - 0.0025, B_R - B_WALL - 0.0055
BASKET_POS = (0.31, 0.07, TOP)
LID_SEAT = B_H - (L_H - L_T)               # the lid's origin (its skirt's lower edge) above the basket's floor when it sits on the rim
LID_POS = (BASKET_POS[0], BASKET_POS[1], TOP + LID_SEAT + 0.0015)


def _basket():
    wall = m.lathe([(B_R - B_WALL, B_WALL), (B_R - B_WALL, B_H), (B_FLANGE, B_H), (B_FLANGE, B_H - 0.002), (B_R, B_H - 0.002), (B_R, 0.0),
                    (0.0, 0.0), (0.0, B_WALL)], 48)
    handles = m.join(*[m.box((HANDLE_W, (HANDLE_SPAN / 2 - B_R), HANDLE_T), (0.0, s * (B_R + (HANDLE_SPAN / 2 - B_R) / 2), B_H - HANDLE_T / 2)) for s in (1, -1)])
    parts = [("mesh", wall, "mesh"), ("handles", handles, "steel"), ("rim", m.lathe([(B_R - B_WALL, B_H - 0.003), (B_FLANGE, B_H - 0.003), (B_FLANGE, B_H), (B_R - B_WALL, B_H)], 48), "steel")]
    cols = [m.col_cylinder("floor", B_R, B_WALL, (0, 0, B_WALL / 2))] + _ring(B_R - B_WALL, B_R, B_WALL, B_H, 16, "wall") + \
        [m.col_box(f"handle{k}", (HANDLE_W, HANDLE_SPAN / 2 - B_R, HANDLE_T), (0.0, s * (B_R + (HANDLE_SPAN / 2 - B_R) / 2), B_H - HANDLE_T / 2)) for k, s in enumerate((1, -1))]
    return parts, cols


def _lid():
    body = m.lathe([(0.0, L_H - L_T), (PLUG_RI, L_H - L_T), (PLUG_RI, 0.0), (PLUG_RO, 0.0), (PLUG_RO, L_H - L_T), (L_R, L_H - L_T), (L_R, L_H), (0.0, L_H)], 48)
    cols = [m.col_cylinder("top", L_R, L_T, (0, 0, L_H - L_T / 2))] + _ring(PLUG_RI, PLUG_RO, 0.0, L_H - L_T, 16, "plug")
    return [("body", body, "lidc")], cols

def _ring(r_in, r_out, z0, z1, n=16, name="w", r_in_top=None, r_out_top=None):
    """Hull pieces of a thin wall of revolution (a glass's side, a dish's rim)."""
    r_in_top = r_in if r_in_top is None else r_in_top
    r_out_top = r_out if r_out_top is None else r_out_top
    cols = []
    for k in range(n):
        th = np.radians(np.linspace(360.0 * k / n, 360.0 * (k + 1) / n, 3))
        pts = [(r * math.cos(t), r * math.sin(t), z) for t in th for r, z in ((r_in, z0), (r_out, z0), (r_in_top, z1), (r_out_top, z1))]
        cols.append(m.col_hull(f"{name}{k}", m.Mesh(np.array(pts, float), np.zeros((1, 3), int))))
    return cols


# -- glass (Libbey Gibraltar 15232 rocks glass): origin at the base's centre --------------------------------------------------
G_H, G_RB, G_RT, G_WALL, G_FLOOR, G_FILL = 0.098, 0.032, 0.0445, 0.003, 0.012, 0.065
glass_body = m.lathe([(0.0, 0.0), (G_RB, 0.0), (G_RT, G_H), (G_RT - G_WALL, G_H), (G_RB - G_WALL + 0.0005, G_FLOOR), (0.0, G_FLOOR)], 40)
_r_at = lambda z: G_RB + (G_RT - G_RB) * z / G_H  # noqa: E731
lemonade = m.lathe([(0.0, G_FLOOR + 0.0005), (_r_at(G_FLOOR) - G_WALL - 0.0005, G_FLOOR + 0.0005), (_r_at(G_FILL) - G_WALL - 0.0005, G_FILL), (0.0, G_FILL)], 40)
glass_cols = [m.col_cylinder("floor", G_RB, G_FLOOR, (0, 0, G_FLOOR / 2))] + _ring(_r_at(G_FLOOR) - G_WALL, _r_at(G_FLOOR), G_FLOOR, G_H, 16, "wall", G_RT - G_WALL, G_RT)

# -- dish of pellets, card, place mat, coaster ---------------------------------------------------------------------------------
DISH_POS = (0.24, -0.04, TOP)
D_R, D_H = 0.055, 0.015
dish_body = m.lathe([(0.0, 0.0), (D_R - 0.004, 0.0), (D_R, D_H), (D_R - 0.002, D_H), (D_R - 0.005, 0.002), (0.0, 0.002)], 40)
dish_cols = [m.col_cylinder("base", D_R - 0.004, 0.002, (0, 0, 0.001))] + _ring(D_R - 0.007, D_R - 0.004, 0.0, D_H, 16, "rim", D_R - 0.002, D_R)
P_R, P_L, P_M = 0.0079, 0.020, 0.006  # dry-ice pellet 16 mm across, 20 mm long, about 6 g
PELLETS = [(DISH_POS[0], DISH_POS[1], TOP + 0.002 + P_R + 0.0005)]  # one pellet, in the middle of the dish
ALONG_X = (0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4))  # a cylinder's axis along x
MAT_POS, MAT_SIZE = (0.52, -0.09, TOP), (0.30, 0.43, 0.002)
COASTER_POS, C_R, C_H = (0.42, -0.12, TOP + MAT_SIZE[2]), 0.05, 0.004
REST_Z = COASTER_POS[2] + C_H  # the glass's base on the coaster
GLASS_POS = (COASTER_POS[0], COASTER_POS[1], REST_Z)

_dots = m.join(*[m.cylinder(0.018, 0.0006, (dx, dy, MAT_SIZE[2] + 0.0003), segments=20) for dx in (-0.10, 0.10) for dy in (-0.16, 0.0, 0.16) if not (dx < 0 and dy == 0.0)])
pellet_mesh = m.cylinder(P_R, P_L, segments=20)
OBJECTS = [
    m.fixed("mat", MAT_POS, [("body", m.box(MAT_SIZE, (0, 0, MAT_SIZE[2] / 2)), "mat"), ("dots", _dots, "matdots")], [m.col_box("c", MAT_SIZE, (0, 0, MAT_SIZE[2] / 2))],
            {"mat": LOOKS["mat"], "matdots": LOOKS["matdots"]}),
    m.fixed("coaster", COASTER_POS, [("body", m.cylinder(C_R, C_H, (0, 0, C_H / 2), segments=48), "cork")], [m.col_cylinder("c", C_R, C_H, (0, 0, C_H / 2))], {"cork": LOOKS["cork"]}),
    m.fixed("dish", DISH_POS, [("body", dish_body, "steel")], dish_cols, {"steel": LOOKS["steel"]}),
    m.fixed("card", (DISH_POS[0] + 0.085, DISH_POS[1], TOP), [("tent", m.prism([(-0.035, -0.012), (0.035, -0.012), (0.035, -0.010), (-0.035, -0.010)], 0.0, 0.035), "card")], [], {"card": LOOKS["card"]}, yaw_deg=90.0),
    m.free("glass", GLASS_POS, [("body", glass_body, "glass"), ("drink", lemonade, "lemonade")], glass_cols, {"glass": LOOKS["glass"], "lemonade": LOOKS["lemonade"]}, mass=0.50, friction=(0.8, 0.7)),
    m.free("basket", BASKET_POS, _basket()[0], _basket()[1], {k: LOOKS[k] for k in ("mesh", "steel")}, mass=0.040, friction=(0.6, 0.5)),
    m.free("lid", LID_POS, _lid()[0], _lid()[1], {"lidc": LOOKS["lidc"]}, mass=0.015, friction=(0.6, 0.5)),
] + [m.free(f"pellet{i}", p, [("body", pellet_mesh, "ice")], [m.col_cylinder("c", P_R, P_L)], {"ice": LOOKS["ice"]}, mass=P_M, start_quat=ALONG_X, friction=(0.5, 0.4)) for i, p in enumerate(PELLETS)]

PELLET_NAMES = [f"pellet{i}" for i in range(len(PELLETS))]
PELLET = PELLET_NAMES[0]


# -- scoring ---------------------------------------------------------------------------------------------------------------------


def pellet_points(pellet) -> np.ndarray:
    """Points round the rims of both of a pellet's end faces, in the world."""
    th = np.linspace(0.0, 2 * math.pi, 8, endpoint=False)
    local = [(P_R * math.cos(t), P_R * math.sin(t), s * P_L / 2) for t in th for s in (-1.0, 1.0)]
    return pellet.to_world(np.array(local))


def lid_on(basket, lid) -> bool:
    """The lid sits on the basket's rim: centred on it within 6 mm, its underside within 4 mm of the rim, tilted under 10 degrees."""
    rel = basket.to_local(lid.pos)[0]
    return bool(np.hypot(rel[0], rel[1]) <= 0.006 and abs(rel[2] - LID_SEAT) <= 0.004 and lid.tilt_deg() < 10.0)


def inside_basket(basket, pellet) -> bool:
    """The whole pellet inside the basket: every rim point within its wall and between its floor and its top (1.5 mm allowed)."""
    q = basket.to_local(pellet_points(pellet))
    return bool(np.all(np.hypot(q[:, 0], q[:, 1]) <= B_R - B_WALL + 0.0015) and np.all(q[:, 2] >= B_WALL - 0.0015) and np.all(q[:, 2] <= B_H + 0.0015))


def in_basket_column(basket, p) -> bool:
    """A point inside the basket's open-topped column: within its wall, above its floor, at any height (a pellet falling into it)."""
    q = basket.to_local(p)[0]
    return bool(np.hypot(q[0], q[1]) <= B_R - B_WALL + 0.0015 and q[2] >= B_WALL - 0.0015)


def in_glass(glass, p, margin: float = 0.0) -> bool:
    """A point inside the glass's bowl, up to `margin` above its rim."""
    local = glass.to_local(p)[0]
    z = local[2]
    return bool(G_FLOOR - 0.002 <= z <= G_H + margin and np.hypot(local[0], local[1]) <= _r_at(min(max(z, 0.0), G_H)) - G_WALL + 0.001)


def on_mat(p) -> bool:
    return bool(abs(p[0] - MAT_POS[0]) <= MAT_SIZE[0] / 2 and abs(p[1] - MAT_POS[1]) <= MAT_SIZE[1] / 2 and p[2] < TOP + 0.04)


class Trial:
    """What the scorer holds across a trial."""

    def reset(self, env=None) -> None:
        self.n = 0
        self.loose = False
        self.spilled = False
        self.hold = 0
        self.score = 0.0
        self.full = False
        if env is not None:
            self.step(env)

    def step(self, env) -> None:
        basket = m.state(env, "basket")
        lid = m.state(env, "lid")
        glass = m.state(env, "glass")
        shut = lid_on(basket, lid)
        pstates = [m.state(env, n) for n in PELLET_NAMES]
        ins = [shut and inside_basket(basket, p) for p in pstates]
        for p, e in zip(pstates, ins):
            if not e and not in_basket_column(basket, p.pos) and (in_glass(glass, p.pos) or on_mat(p.pos)):
                self.loose = True
        base = glass.pos
        on_coaster = bool(np.hypot(*(base[:2] - np.array(COASTER_POS[:2]))) <= C_R - G_RB and abs(base[2] - REST_Z) < 0.004)
        self.spilled = self.spilled or glass.tilt_deg() > 20.0 or not on_coaster
        floor = basket.to_world(np.array([[0.0, 0.0, 0.0]]))[0]
        basket_in = bool(in_glass(glass, floor + [0, 0, 0.01]) and np.hypot(*(glass.to_local(floor)[0][:2])) <= 0.015)
        caught = any(ins)
        tips = m.fingertips(env)
        gap_glass = float(np.min(np.linalg.norm((tips - glass.pos)[:, :2], axis=1))) - G_RT
        gap_lid = float(np.min(np.linalg.norm((tips - lid.pos)[:, :2], axis=1))) - L_R
        upright = glass.tilt_deg() < 5.0
        still = bool(np.linalg.norm(glass.lin_vel) < 0.01 and np.linalg.norm(basket.lin_vel) < 0.01)
        clear = bool(min(gap_glass, gap_lid) >= 0.03)
        end_ok = bool(caught and basket_in and upright and still and clear)
        self.hold = self.hold + 1 if end_ok else 0
        if self.loose or self.spilled:
            self.score = 0.0
        elif caught and basket_in and self.hold >= 20:
            self.score = 1.0
        elif caught and basket_in:
            self.score = 0.5
        elif caught:
            self.score = 0.25
        else:
            self.score = 0.0
        self.full = self.score == 1.0
        self.n += 1
        if TRACE and (self.n % 10 == 0 or self.n < 6):
            p1 = pstates[0].pos
            if self.n % 100 == 0:
                print(f"[dry] pellets {[np.round(p.pos - [0, 0, TOP], 3).tolist() for p in pstates]}", flush=True)
            print(f"[dry] step {self.n}: basket {np.round(basket.pos - [0, 0, TOP], 3)} tilt {basket.tilt_deg():.1f} lid {np.round(lid.pos - [0, 0, TOP], 3)} shut {shut} "
                  f"p1 {np.round(p1 - [0, 0, TOP], 3)} in {ins} basket_in {basket_in} glass {np.round(base - [0, 0, TOP], 3)} gtilt {glass.tilt_deg():.1f} "
                  f"loose {self.loose} spill {self.spilled} clear {gap_glass:.3f}/{gap_lid:.3f} score {self.score}", flush=True)


TRIAL = Trial()
TRIAL.reset()
m.on_step(TRIAL.step)
m.on_reset(TRIAL.reset)


def success(objects) -> bool:
    steptime.tick()  # the harness calls success() once per control step
    return TRIAL.full


def score(objects) -> float:
    return TRIAL.score


# -- demo ----------------------------------------------------------------------------------------------------------------------------


LID_REST = (0.18, 0.06)  # where the left hand sets the lid down, plug down, for the right hand to take later


def demo_script(ctx):
    """Modes: full; none, the arms wait; failure, the pellet dropped straight into the glass.
    Full: the left hand lifts the lid off and sets it aside; the right hangs the empty basket in the glass by one handle, drops a pellet
    into it, then takes the lid and sets it back in the basket's mouth."""
    l, r = ctx.left, ctx.right
    mode = ctx.mode
    if mode == "none":
        yield from l.wait(3.0)
        return
    pellet = lambda: ctx.pos(PELLET)  # noqa: E731
    basket = lambda: ctx.state("basket")  # noqa: E731
    lid_grip_h = L_H - L_T / 2   # the fingertips on the lid's disc, across its 76 mm
    if mode == "failure":
        yield from r.move(pellet() + [0, 0, 0.08], grip=d.down(0.0), finger=-0.0105, t=2.5, label="right: over a pellet, fingers opened just past its 16 mm")
        yield from r.move(pellet(), t=1.0)
        yield from r.close(2 * P_R, force=8.0)
        yield from r.move(r.at() + [0, 0, 0.10], t=1.0)
        yield from r.move(ctx.pos("glass") + [0, 0, G_H + 0.05], t=2.5, label="right: over the glass")
        yield from r.move(r.at(), finger=-0.016, t=0.4, label="right: drop it in")
        yield from r.move((0.20, -0.30, TOP + 0.20), t=2.0)
        yield from r.wait(2.0)
        return
    # 1. left: lid off, set down aside
    yield from l.move(ctx.pos("lid") + [0, 0, lid_grip_h + 0.06], grip=d.down(-90.0), finger=d.FINGER_OPEN, t=2.5, label="left: over the lid")
    yield from l.servo(lambda *_: ctx.pos("lid") + [0, 0, lid_grip_h], timeout=1.5, label="left: fingers round the lid")
    yield from l.close(2 * L_R, force=10.0, label="left: hold the lid")
    yield from l.move(l.at() + [0, 0, 0.06], t=1.0, label="left: lift the lid off")
    off = l.at() - ctx.pos("lid")
    rest = np.array([LID_REST[0], LID_REST[1], TOP + 0.002]) + off
    yield from l.move(rest + [0, 0, 0.03], t=1.5, label="left: carry the lid aside")
    yield from l.move(rest, t=0.8, label="left: set it down")
    yield from l.open(t=0.5)
    yield from l.move(l.at() + [0, 0, 0.06], t=0.6)
    yield from l.move((0.18, 0.28, TOP + 0.20), t=1.5, label="left: away")
    # 2. right: the empty basket into the glass by one handle's tip
    hold = lambda *_: basket().to_world(np.array([[0.0, -(HANDLE_SPAN / 2 - 0.009), B_H - HANDLE_T / 2]]))[0]  # noqa: E731
    yield from r.move(r.at() + [0, 0, 0.12], t=1.5, label="right: straight up from the ready pose")
    yield from r.move((0.22, -0.20, TOP + 0.24), grip=d.down(90.0), finger=d.FINGER_OPEN, t=2.0, label="right: up, high over the dish")
    yield from r.move(hold() + [0, 0, 0.07], t=2.0, label="right: over the basket's handle")
    yield from r.servo(hold, timeout=1.5, label="right: fingers either side of the handle")
    yield from r.close(HANDLE_W, force=12.0, label="right: pinch the handle across its width")
    yield from r.move(r.at() + [0, 0, 0.11], t=2.0, label="right: lift it")
    g = ctx.pos("glass")
    floor_at = lambda h: g + [0.0, 0.0, G_H + h + HANDLE_T - B_H]  # noqa: E731  (the basket's floor, with its handles h above the rim)
    yield from r.move(r.at() + (floor_at(B_H * 0 + 0.05) - basket().pos), t=3.0, label="right: over the glass")
    yield from r.servo(lambda *_: floor_at(0.03), read=lambda *_: basket().pos, timeout=1.5)
    yield from r.servo(lambda *_: floor_at(0.004), read=lambda *_: basket().pos, timeout=2.0, label="right: lower it in, handles onto the rim")
    yield from r.open(t=0.5, label="right: let go")
    yield from r.move(r.at() + [0, 0, 0.05], t=1.0, label="right: up")
    # 3. right: a pellet into the basket, not the glass
    yield from r.move(pellet() + [0, 0, 0.08], grip=d.down(0.0), finger=-0.0105, t=2.5, label="right: over a pellet, fingers opened just past its 16 mm")
    yield from r.move(pellet(), t=1.0, label="right: down to the pellet")
    yield from r.close(2 * P_R, force=8.0, label="right: pinch the pellet")
    yield from r.move(r.at() + [0, 0, 0.12], t=1.2, label="right: lift it")
    over = lambda *_: basket().pos + [0.0, 0.0, B_H + P_R + 0.02]  # noqa: E731
    yield from r.move(over(), t=2.5, label="right: over the basket in the glass")
    yield from r.servo(over, timeout=1.5, read=lambda *_: pellet())
    yield from r.move(r.at(), finger=-0.016, t=0.4, label="right: drop the pellet into the basket")
    yield from r.move(r.at() + [0, 0, 0.04], finger=d.FINGER_OPEN, t=0.8, label="right: up")
    if TRACE:
        print(f"[dry] loaded: pellet {np.round(basket().to_local(pellet())[0], 4)} inside {inside_basket(basket(), ctx.state(PELLET))}", flush=True)
    # 4. right: the lid back into the basket's mouth
    yield from r.move(ctx.pos("lid") + [0, 0, lid_grip_h + 0.06], grip=d.down(90.0), finger=d.FINGER_OPEN, t=2.5, label="right: over the lid")
    yield from r.servo(lambda *_: ctx.pos("lid") + [0, 0, lid_grip_h], timeout=1.5, label="right: fingers round the lid")
    yield from r.close(2 * L_R, force=10.0, label="right: hold the lid")
    yield from r.move(r.at() + [0, 0, 0.10], t=1.0, label="right: lift it")
    off = r.at() - ctx.pos("lid")
    seat = basket().pos + [0.0, 0.0, LID_SEAT + 0.003] + off
    yield from r.move(seat + [0, 0, 0.03], t=2.0, label="right: lid over the basket")
    yield from r.wait(0.5)
    off2 = ctx.pos("lid") - (basket().pos + [0.0, 0.0, LID_SEAT + 0.033])
    yield from r.move(r.at() - [off2[0], off2[1], 0.0], t=0.6, label="right: centre it over the mouth")
    yield from r.move(r.at() - [0, 0, 0.03], t=1.0, label="right: lower the lid into the mouth")
    yield from r.open(t=0.5, label="right: let go")
    yield from r.move(r.at() + [0, 0, 0.05], t=1.0, label="right: up and clear")
    if TRACE:
        print(f"[dry] shut: lid_on {lid_on(basket(), ctx.state('lid'))} inside {inside_basket(basket(), ctx.state(PELLET))}", flush=True)
    yield from r.move((0.20, -0.30, TOP + 0.20), grip=d.down(0.0), t=1.5, label="right: away")
    yield from r.wait(2.5)
