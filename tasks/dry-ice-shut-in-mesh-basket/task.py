"""A child's drink made to fog with dry ice, the pellet shut inside a screw-top mesh tea infuser in the glass, never loose in it.

On the station table: a small stainless dish holding a dry-ice pellet, a screw-top stainless mesh tea infuser with its cap beside it,
and the child's place: a place mat with a cork coaster and a rocks glass of lemonade on it. The pellet goes into the infuser,
the cap is screwed on, and the closed infuser goes down into the glass.

    python demokit.py task.py OUT --mode full        the scripted demo
    python demokit.py task.py OUT --mode failure     the pellet dropped straight into the glass (scores 0)
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

import numpy as np
from scipy.spatial import ConvexHull

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

# -- screw-top mesh tea infuser (a 1.5 x 2.2 in threaded-lid capsule): part frames at the capsule's floor centre and the cap's rim -------
B_R, B_H, B_WALL = 0.019, 0.046, 0.002      # capsule 38 mm across, 46 mm to the top of its threaded collar (estimate within the 2.2 in listed height)
CAP_R, CAP_H, CAP_T = 0.0245, 0.012, 0.002  # cap 49 mm across, 12 mm tall, its skirt screws over the collar (estimate)
CAP_SKIRT_IN = B_R + 0.003                  # 3 mm clear of the collar: the thread itself is not drawn
CAP_SEAT = B_H - (CAP_H - CAP_T)            # the cap's origin (its skirt's lower edge) above the capsule's floor when screwed down
SCREW_TURNS = 1.0                           # a full turn while seated closes the thread (estimate; the real cap takes about 1.5 turns)
CLAMP_N = 3.0                               # the closed thread holds the cap down with 3 N (model of the thread's grip)
HANG_N = 0.6                                # once the closed cap is lifted, the thread carries the infuser with 0.6 N (twice its weight with the pellet)
PULL_N = 1.0                                # the scorer's pull test on the cap at the end
CAPSULE_POS = (0.30, 0.06, TOP)
CAP_POS = (0.22, 0.16, TOP)                 # the cap lies on the left, open side down


def _capsule():
    body = m.lathe([(0.0, 0.0), (B_R, 0.0), (B_R, B_H), (B_R - B_WALL, B_H), (B_R - B_WALL, B_WALL), (0.0, B_WALL)], 40)
    collar = m.lathe([(B_R - B_WALL, B_H - 0.008), (B_R + 0.0008, B_H - 0.008), (B_R + 0.0008, B_H), (B_R - B_WALL, B_H)], 40)
    parts = [("mesh", body, "mesh"), ("collar", collar, "steel")]
    cols = [m.col_cylinder("floor", B_R, B_WALL, (0, 0, B_WALL / 2))] + _ring(B_R - B_WALL, B_R, B_WALL, B_H, 16, "wall")
    return parts, cols


def _cap():
    body = m.lathe([(0.0, CAP_H - CAP_T), (CAP_SKIRT_IN, CAP_H - CAP_T), (CAP_SKIRT_IN, 0.0), (CAP_R, 0.0), (CAP_R, CAP_H), (0.0, CAP_H)], 48)
    ridges = m.join(*[m.box((0.0015, 0.004, CAP_H - 0.002), (CAP_R * math.cos(t), CAP_R * math.sin(t), CAP_H / 2)).turned(0.0) for t in np.linspace(0, 2 * math.pi, 24, endpoint=False)])
    cols = [m.col_cylinder("top", CAP_R, CAP_T, (0, 0, CAP_H - CAP_T / 2))] + _ring(CAP_SKIRT_IN, CAP_R, 0.0, CAP_H - CAP_T, 16, "skirt")
    return [("body", body, "steel"), ("grip", ridges, "steel")], cols


def _ring(r_in, r_out, z0, z1, n=16, name="w", r_in_top=None, r_out_top=None):
    """Hull pieces of a thin wall of revolution (a glass's side, a dish's rim)."""
    r_in_top = r_in if r_in_top is None else r_in_top
    r_out_top = r_out if r_out_top is None else r_out_top
    cols = []
    for k in range(n):
        th = np.radians(np.linspace(360.0 * k / n, 360.0 * (k + 1) / n, 3))
        pts = [(r * math.cos(t), r * math.sin(t), z) for t in th for r, z in ((r_in, z0), (r_out, z0), (r_in_top, z1), (r_out_top, z1))]
        pts = np.array(pts, float)
        cols.append(m.col_hull(f"{name}{k}", m.Mesh(pts, ConvexHull(pts).simplices.astype(int))))  # real faces: a hull of one empty face collides with nothing
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
MAT_POS, MAT_SIZE = (0.48, -0.12, TOP), (0.30, 0.43, 0.002)
COASTER_POS, C_R, C_H = (0.40, -0.20, TOP + MAT_SIZE[2]), 0.05, 0.004
REST_Z = COASTER_POS[2] + C_H  # the glass's base on the coaster
GLASS_POS = (COASTER_POS[0], COASTER_POS[1], REST_Z)

_dots = m.join(*[m.cylinder(0.018, 0.0006, (dx, dy, MAT_SIZE[2] + 0.0003), segments=20) for dx in (-0.10, 0.10) for dy in (-0.16, 0.0, 0.16) if not (dx < 0 and dy == 0.0)])
pellet_mesh = m.cylinder(P_R, P_L, segments=20)
OBJECTS = [
    m.fixed("mat", MAT_POS, [("body", m.box(MAT_SIZE, (0, 0, MAT_SIZE[2] / 2)), "mat"), ("dots", _dots, "matdots")], [m.col_box("c", MAT_SIZE, (0, 0, MAT_SIZE[2] / 2))],
            {"mat": LOOKS["mat"], "matdots": LOOKS["matdots"]}),
    m.fixed("coaster", COASTER_POS, [("body", m.cylinder(C_R, C_H, (0, 0, C_H / 2), segments=48), "cork")], [m.col_cylinder("c", C_R, C_H, (0, 0, C_H / 2))], {"cork": LOOKS["cork"]}),
    m.fixed("dish", DISH_POS, [("body", dish_body, "steel")], dish_cols, {"steel": LOOKS["steel"]}),
    m.fixed("card", (DISH_POS[0] + 0.075, DISH_POS[1], TOP), [("tent", m.prism([(-0.035, -0.012), (0.035, -0.012), (0.035, -0.010), (-0.035, -0.010)], 0.0, 0.035), "card")], [], {"card": LOOKS["card"]}, yaw_deg=90.0),
    m.free("glass", GLASS_POS, [("body", glass_body, "glass"), ("drink", lemonade, "lemonade")], glass_cols, {"glass": LOOKS["glass"], "lemonade": LOOKS["lemonade"]}, mass=0.50, friction=(0.8, 0.7)),
    m.free("capsule", CAPSULE_POS, _capsule()[0], _capsule()[1], {k: LOOKS[k] for k in ("mesh", "steel")}, mass=0.025, friction=(0.6, 0.5)),
    m.free("cap", CAP_POS, _cap()[0], _cap()[1], {"steel": LOOKS["steel"]}, mass=0.012, friction=(0.7, 0.6)),
] + [m.free(f"pellet{i}", p, [("body", pellet_mesh, "ice")], [m.col_cylinder("c", P_R, P_L)], {"ice": LOOKS["ice"]}, mass=P_M, start_quat=ALONG_X, friction=(0.5, 0.4)) for i, p in enumerate(PELLETS)]

PELLET_NAMES = [f"pellet{i}" for i in range(len(PELLETS))]
PELLET = PELLET_NAMES[0]


# -- scoring ---------------------------------------------------------------------------------------------------------------------


def pellet_points(pellet) -> np.ndarray:
    """Points round the rims of both of a pellet's end faces, in the world."""
    th = np.linspace(0.0, 2 * math.pi, 8, endpoint=False)
    local = [(P_R * math.cos(t), P_R * math.sin(t), s * P_L / 2) for t in th for s in (-1.0, 1.0)]
    return pellet.to_world(np.array(local))


def cap_seated(capsule, cap) -> bool:
    """The cap sits on the collar: its axis within 3 mm of the capsule's, its skirt's edge within 3 mm of the screwed-down height, tilted under 10 degrees from it."""
    rel = capsule.to_local(cap.pos)[0]
    tilt = math.degrees(math.acos(np.clip(float(capsule.rotation()[:, 2] @ cap.rotation()[:, 2]), -1.0, 1.0)))
    return bool(np.hypot(rel[0], rel[1]) <= 0.003 and abs(rel[2] - CAP_SEAT) <= 0.003 and tilt < 10.0)


def cap_yaw(capsule, cap) -> float:
    """The cap's turn about the capsule's axis, in degrees."""
    x = capsule.rotation().T @ cap.rotation()[:, 0]
    return math.degrees(math.atan2(x[1], x[0]))


def inside_capsule(capsule, pellet) -> bool:
    """The whole pellet inside the capsule: every rim point within its wall, above its floor and below its top (1.5 mm allowed)."""
    q = capsule.to_local(pellet_points(pellet))
    return bool(np.all(np.hypot(q[:, 0], q[:, 1]) <= B_R - B_WALL + 0.0015) and np.all(q[:, 2] >= B_WALL - 0.0015) and np.all(q[:, 2] <= B_H + 0.0015))


def in_capsule_column(capsule, p) -> bool:
    """A point inside the capsule's open-topped column: within its wall, above its floor, at any height (a pellet falling into it)."""
    q = capsule.to_local(p)[0]
    return bool(np.hypot(q[0], q[1]) <= B_R - B_WALL + 0.0015 and q[2] >= B_WALL - 0.0015)


def in_glass(glass, p, margin: float = 0.0) -> bool:
    """A point inside the glass's bowl, up to `margin` above its rim."""
    local = glass.to_local(p)[0]
    z = local[2]
    return bool(G_FLOOR - 0.002 <= z <= G_H + margin and np.hypot(local[0], local[1]) <= _r_at(min(max(z, 0.0), G_H)) - G_WALL + 0.001)


def on_mat(p) -> bool:
    return bool(abs(p[0] - MAT_POS[0]) <= MAT_SIZE[0] / 2 and abs(p[1] - MAT_POS[1]) <= MAT_SIZE[1] / 2 and p[2] < TOP + 0.04)


class Trial:
    """What the scorer holds across a trial. The thread: while the cap is seated, its turns in the tightening sense (clockwise seen from
    above) are counted; a full turn closes it, and from then on it holds the cap down with CLAMP_N until it is turned back. The pull test:
    once everything else has held still for 1 s, the cap is pulled up with PULL_N for 0.5 s and must stay seated."""

    def reset(self, env=None) -> None:
        self.n = 0
        self.loose = False
        self.spilled = False
        self.turned = 0.0
        self.last_yaw = None
        self.closed = False
        self.clamping = False
        self.hung = False
        self.hold = 0
        self.pull = 0
        self.pulled_ok = False
        self.score = 0.0
        self.full = False
        if env is not None:
            self.step(env)

    def _clamp(self, env, on: bool, up: float = 0.0) -> None:
        """The closed thread: it holds the cap down on the collar with CLAMP_N. Once the closed cap is lifted (by a hand, more than 1 mm above
        its seat), it also holds the infuser up to the cap with HANG_N, so the closed infuser hangs from its cap (that pair of forces is equal
        and opposite; the 3 N hold-down acts on the cap alone). The pull test pulls the cap up and the infuser down with `up` (a judge
        holding the infuser and pulling the cap); it runs only on a closed cap, whose hold-down exceeds it, so in the simulation it re-checks
        that the cap is closed and seated rather than testing a modelled thread strength."""
        if on or up or self.clamping:
            hang = HANG_N if on and self.hung else 0.0
            m.push(env, "cap", (0.0, 0.0, (-CLAMP_N if on else 0.0) - hang + up))
            m.push(env, "capsule", (0.0, 0.0, hang - up))
        self.clamping = on or bool(up)

    def step(self, env) -> None:
        capsule = m.state(env, "capsule")
        cap = m.state(env, "cap")
        glass = m.state(env, "glass")
        seated = cap_seated(capsule, cap)
        yaw = cap_yaw(capsule, cap)
        if seated and self.last_yaw is not None:
            dy = (yaw - self.last_yaw + 180.0) % 360.0 - 180.0
            self.turned = min(max(self.turned - dy, 0.0), 720.0)  # clockwise seen from above tightens
        elif not seated:
            self.turned = 0.0  # off its seat the cap is off the thread: a closed cap pulled off and set back on is open again
            self.hung = False
        self.last_yaw = yaw
        self.closed = bool(self.turned >= 360.0 * SCREW_TURNS)
        shut = bool(self.closed and seated)
        if shut and capsule.to_local(cap.pos)[0][2] > CAP_SEAT + 0.0005:
            self.hung = True  # the closed cap has been lifted: from now on the thread also carries the infuser
        pstates = [m.state(env, n) for n in PELLET_NAMES]
        ins = [shut and inside_capsule(capsule, p) for p in pstates]
        for p, e in zip(pstates, ins):
            if not e and not in_capsule_column(capsule, p.pos) and (in_glass(glass, p.pos) or on_mat(p.pos)):
                self.loose = True
        base = glass.pos
        on_coaster = bool(np.hypot(*(base[:2] - np.array(COASTER_POS[:2]))) <= C_R - G_RB and abs(base[2] - REST_Z) < 0.004)
        self.spilled = self.spilled or glass.tilt_deg() > 20.0 or not on_coaster
        floor = capsule.pos
        capsule_in = bool(in_glass(glass, floor + [0, 0, 0.005]) and glass.to_local(floor)[0][2] + B_H + (CAP_H - CAP_T) <= G_H)
        caught = any(ins)
        tips = m.fingertips(env)
        gap_glass = float(np.min(np.linalg.norm((tips - glass.pos)[:, :2], axis=1))) - G_RT
        upright = glass.tilt_deg() < 5.0
        still = bool(np.linalg.norm(glass.lin_vel) < 0.01 and np.linalg.norm(capsule.lin_vel) < 0.01)
        clear = bool(gap_glass >= 0.03)
        end_ok = bool(caught and capsule_in and upright and still and clear)
        self.hold = self.hold + 1 if end_ok or self.pull else 0
        if self.hold >= 20 and not self.pulled_ok:  # the pull test, after 1 s of everything held still
            self.pull += 1
            if not seated:
                self.pull, self.hold = 0, 0
            elif self.pull >= 10:
                self.pulled_ok = True
        self._clamp(env, self.closed and seated, PULL_N if 0 < self.pull < 10 and not self.pulled_ok else 0.0)
        if self.loose or self.spilled:
            self.score = 0.0
        elif caught and capsule_in and self.pulled_ok and end_ok:
            self.score = 1.0
        elif caught and capsule_in:
            self.score = 0.5
        elif caught:
            self.score = 0.25
        else:
            self.score = 0.0
        self.full = self.score == 1.0
        self.n += 1
        if TRACE and (self.n % 10 == 0 or self.n < 6):
            p1 = pstates[0].pos
            print(f"[dry] step {self.n}: capsule {np.round(capsule.pos - [0, 0, TOP], 3)} tilt {capsule.tilt_deg():.1f} cap {np.round(cap.pos - [0, 0, TOP], 3)} seated {seated} "
                  f"turned {self.turned:.0f} closed {self.closed} pull {self.pull} ok {self.pulled_ok} p1 {np.round(p1 - [0, 0, TOP], 3)} in {ins} capsule_in {capsule_in} "
                  f"gtilt {glass.tilt_deg():.1f} loose {self.loose} spill {self.spilled} clear {gap_glass:.3f} score {self.score}", flush=True)


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


def demo_script(ctx):
    """Modes: full; loose_cap, the cap set on but not screwed; none, the arms wait; failure, the pellet dropped straight into the glass.
    Full: the right hand drops the pellet into the open capsule; the left hand sets the cap on the collar and screws it down in three
    150-degree strokes (a little over a turn); then the right hand lifts the closed capsule by its cap and lowers it into the glass."""
    l, r = ctx.left, ctx.right
    mode = ctx.mode
    if mode == "none":
        yield from l.wait(3.0)
        return
    pellet = lambda: ctx.pos(PELLET)  # noqa: E731
    capsule = lambda: ctx.state("capsule")  # noqa: E731
    yield from r.move(r.at() + [0, 0, 0.12], t=1.5, label="right: straight up from the ready pose")
    yield from r.move(pellet() + [0, 0, 0.08], grip=d.down(0.0), finger=-0.0105, t=2.5, label="right: over the pellet, fingers opened just past its 16 mm")
    yield from r.move(pellet(), t=1.0, label="right: down to the pellet")
    yield from r.close(2 * P_R, force=8.0, label="right: pinch the pellet")
    yield from r.move(r.at() + [0, 0, 0.10], t=1.0, label="right: lift it")
    if mode == "failure":
        yield from r.move(ctx.pos("glass") + [0, 0, G_H + 0.05], t=2.5, label="right: over the glass")
        yield from r.move(r.at(), finger=-0.016, t=0.4, label="right: drop it in")
        yield from r.move((0.20, -0.30, TOP + 0.20), t=2.0)
        yield from r.wait(2.0)
        return
    over = lambda *_: capsule().pos + [0.0, 0.0, B_H + P_R + 0.02]  # noqa: E731
    yield from r.move(over(), grip=d.down(90.0), t=2.5, label="right: over the open capsule")
    yield from r.servo(over, timeout=1.5, read=lambda *_: pellet())
    yield from r.move(r.at(), finger=-0.016, t=0.4, label="right: drop the pellet into the capsule")
    yield from r.move(r.at() + [0, 0, 0.04], finger=d.FINGER_OPEN, t=0.8, label="right: up")
    if TRACE:
        print(f"[dry] loaded: pellet {np.round(capsule().to_local(pellet())[0], 4)} inside {inside_capsule(capsule(), ctx.state(PELLET))}", flush=True)
    yield from r.move(r.at() + [0, 0, 0.04], t=0.6, label="right: up")
    yield from r.move((0.22, -0.26, TOP + 0.20), grip=d.down(0.0), t=1.5, label="right: out of the way")
    # left: the cap, on to the collar
    cap_grip_h = CAP_H - 0.005
    yield from l.move(ctx.pos("cap") + [0, 0, cap_grip_h + 0.06], grip=d.down(0.0), finger=d.FINGER_OPEN, t=2.0, label="left: over the cap")
    yield from l.servo(lambda *_: ctx.pos("cap") + [0, 0, cap_grip_h], timeout=1.5, label="left: fingers round the cap")
    yield from l.close(2 * CAP_R, force=10.0, label="left: hold the cap")
    yield from l.move(l.at() + [0, 0, 0.08], t=1.0, label="left: lift it")
    off = l.at() - ctx.pos("cap")
    above = lambda: capsule().pos + [0.0, 0.0, CAP_SEAT + 0.012]  # noqa: E731  (the cap's skirt 12 mm above its seat)
    yield from l.move(above() + off, t=2.0, label="left: cap over the infuser")
    for _ in range(3):  # centre it from the cap's measured pose
        yield from l.wait(0.4)
        miss = ctx.pos("cap") - above()
        yield from l.move(l.at() - [miss[0], miss[1], 0.0], t=0.6, label="left: centre the cap on the collar")
    yield from l.move(l.at() - [0, 0, 0.0115], t=1.2, label="left: cap down onto the collar")
    yield from l.open(t=0.4, label="left: let go of the cap")
    yield from l.move(l.at(), grip=d.down(45.0), t=0.8, label="left: turn the open hand to the stroke's start")
    # screw: strokes of 150 degrees clockwise seen from above, re-gripping between them
    for k in range(3 if mode != "loose_cap" else 0):
        yield from l.close(2 * CAP_R, force=10.0, t=0.4, label="left: hold the cap")
        yield from l.turn(-150.0, t=2.0, label=f"left: screw the cap, stroke {k + 1}")
        yield from l.open(t=0.4, label="left: let go of the cap")
        if k < 2:
            yield from l.move(l.at(), grip=d.down(45.0), t=1.0, label="left: turn back, open")
    yield from l.move(l.at() + [0, 0, 0.06], t=0.8, label="left: up")
    if TRACE:
        print(f"[dry] screwed: turned {TRIAL.turned:.0f} closed {TRIAL.closed} inside {inside_capsule(capsule(), ctx.state(PELLET))}", flush=True)
    yield from l.move((0.18, 0.28, TOP + 0.20), grip=d.down(0.0), t=1.5, label="left: away")
    # right: the closed capsule into the glass
    top = lambda *_: ctx.pos("cap") + [0.0, 0.0, CAP_H - 0.005]  # noqa: E731  (the closed infuser is carried by its cap)
    yield from r.move(top() + [0, 0, 0.08], grip=d.down(0.0), finger=d.FINGER_OPEN, t=2.0, label="right: over the closed infuser")
    yield from r.servo(top, timeout=1.5, label="right: fingers round the cap")
    yield from r.close(2 * CAP_R, force=10.0, label="right: hold the infuser by its cap")
    yield from r.move(r.at() + [0, 0, 0.02], t=1.5, label="right: lift the infuser gently: the thread takes its weight")
    yield from r.move(r.at() + [0, 0, 0.10], t=1.2, label="right: lift it clear")
    g = ctx.pos("glass")
    yield from r.move(r.at() + (g + [0, 0, G_H + 0.025] - capsule().pos), t=3.0, label="right: over the glass")
    yield from r.servo(lambda *_: g + [0, 0, G_H + 0.006], read=lambda *_: capsule().pos, timeout=2.0, label="right: capsule's base just above the rim, over the middle")
    yield from r.servo(lambda *_: g + [0, 0, G_FLOOR + 0.05], read=lambda *_: capsule().pos, timeout=2.0, label="right: lower it into the glass, fingers clear of the wall")
    yield from r.open(t=0.5, label="right: let it go: it sinks to the glass's floor")
    yield from r.move(r.at() + [0, 0, 0.10], t=1.2, label="right: up and clear")
    yield from r.move((0.20, -0.30, TOP + 0.20), grip=d.down(0.0), t=1.5, label="right: away")
    yield from r.wait(3.0)
