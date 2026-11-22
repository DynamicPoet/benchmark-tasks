"""On the station table: a just made-up baby's bottle standing beside a kettle and an opened formula tin, a stainless bowl of iced
water, and a toddler doll seated in a high-chair seat with its tray in front of her.

The feed is too hot to give. The robot stands the full bottle upright in the iced water (the water stays below the collar), waits
until the feed has cooled to feeding temperature, then sets it upright on the taped square of the baby's tray and lets go.

Feed temperature (a lumped model, stated as an engineering estimate): every control step
    T <- Tw + (T - Tw) exp(-dt / tau_w)   while the bottle stands in the water, with tau_w from the wetted milk column, else
    T <- Ta + (T - Ta) exp(-dt / tau_a)   in air,
tau_w = C / (U A_wet), tau_a = C / (h_a A_bottle). The working is in the README.

    python demokit.py task.py OUT --no-record --mode full,none,hot,brief,leave     debug: every mode in one session
    python demokit.py task.py OUT                                                  final video
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import demokit as d  # noqa: E402
import gatekit as gk  # noqa: E402
import mechkit as m  # noqa: E402
import steptime  # noqa: E402

SCENE = "station"
INSTRUCTION = "Give the baby her bottle: set it upright on the taped square of her tray, then let go and move your hands clear. Leave the bowl where it stands."
SUCCESS_HOLD_S = 0.0  # the scorer holds the end state for 1 s itself
SCORE_OVER = "end"
TOP = d.TABLE_TOP
DT = 1.0 / 20.0

# -- the bottle (Philips Avent Natural Response 260 ml, 7.1 cm across, 16.6 cm tall with its cap) --------------------------------------------
B_R = 0.0355          # outer radius of the body
B_WALL = 0.0012       # polypropylene wall
B_BASE = 0.003        # the milk's floor above the bottle's foot
MILK_ML = 240.0
MILK_R = B_R - B_WALL
MILK_H = MILK_ML * 1e-6 / (math.pi * MILK_R ** 2)   # 0.068 m
COLLAR_Z = 0.110      # the collar's lower edge above the foot (an estimate from the published 16.6 cm height)
B_MASS = 0.28         # 240 g of feed and about 40 g of bottle

# -- the bowl (stainless, 20 cm across the rim, 9 cm deep, flat floor 12 cm across) ---------------------------------------------------------------
W_R0, W_R1, W_H = 0.060, 0.100, 0.090
W_FLOOR = 0.004
WATER = 0.060         # iced water 6 cm deep above the floor
BOWL_MASS = 0.35 + 1.0  # bowl and about 1 L of water


def bowl_r(z: float) -> float:
    """The bowl's inner radius at height z above its foot."""
    return W_R0 - 0.002 + (W_R1 - W_R0) * z / W_H


# -- temperatures and the model's constants (engineering estimates, see the README) ---------------------------------------------------------------
T0 = 50.0             # the feed at reset
T_WATER = 5.0         # iced water, kept there by the ice
T_AIR = 22.0
T_SAFE, T_LOW = 37.0, 30.0
C_FEED = 0.240 * 4180.0 + 0.040 * 1900.0     # J/K, feed and bottle
U_WET = 114.0         # W/m2K, milk -> 1.2 mm PP wall -> still water (inside and outside natural convection, about 600 W/m2K each, and the wall's 183)
H_AIR = 10.0          # W/m2K, natural convection and radiation to room air
A_BOTTLE = 2 * math.pi * B_R * (MILK_H + B_BASE) + math.pi * B_R ** 2


def tau_water(wet: float) -> float:
    """Seconds, for `wet` metres of the milk column below the water line."""
    area = 2 * math.pi * MILK_R * wet + math.pi * MILK_R ** 2
    return C_FEED / (U_WET * area)


TAU_AIR = C_FEED / (H_AIR * A_BOTTLE)

# -- layout (world: x forward, y to the robot's left; heights above the table top) ---------------------------------------------------------------
BOTTLE_AT = (0.26, -0.24, TOP + 0.001)
BOWL_AT = (0.19, -0.08, TOP + 0.001)
KETTLE_AT = (0.32, 0.24, TOP)
TIN_AT = (0.18, 0.23, TOP)
TRAY_TOP = 0.070
TRAY = (0.28, 0.48, -0.53, -0.14)      # x0, x1, y0, y1
SQUARE = (0.35, -0.30)                 # the taped 10 cm square's centre
SQ_HALF = 0.05
SEAT_X = (0.42, 0.64)
SEAT_Y = (-0.51, -0.11)
DOLL_X = 0.555
HEAD = np.array([DOLL_X - 0.025, -0.31, TOP + 0.33])
HEAD_R = 0.065
HANDS_AT = [np.array([0.43, -0.42, TOP + TRAY_TOP + 0.018]), np.array([0.43, -0.20, TOP + TRAY_TOP + 0.018])]
HAND_R = 0.022

L = {
    "pp": m.Look(m.rgb(0.93, 0.95, 0.97), roughness=0.2, opacity=0.35),
    "milk": m.Look(m.rgb(0.96, 0.93, 0.84), roughness=0.5),
    "collar": m.Look(m.rgb(0.80, 0.86, 0.90), roughness=0.3),
    "teat": m.Look(m.rgb(0.86, 0.72, 0.55), roughness=0.5, opacity=0.9),
    "cap": m.Look(m.rgb(0.92, 0.95, 0.98), roughness=0.15, opacity=0.4),
    "steel": m.Look(m.rgb(0.78, 0.79, 0.80), metallic=0.9, roughness=0.25),
    "water": m.Look(m.rgb(0.45, 0.68, 0.90), roughness=0.05, opacity=0.5),
    "ice": m.Look(m.rgb(0.90, 0.96, 1.0), roughness=0.1, opacity=0.75),
    "white": m.Look(m.rgb(0.95, 0.95, 0.94), roughness=0.45),
    "dark": m.Look(m.rgb(0.12, 0.12, 0.13), roughness=0.5),
    "light": m.Look(m.rgb(0.3, 0.6, 1.0), emissive=m.rgb(0.2, 0.5, 1.0)),
    "steam": m.Look(m.rgb(0.98, 0.98, 0.98), roughness=0.9, opacity=0.2),
    "tin": m.Look(m.rgb(0.55, 0.35, 0.65), roughness=0.4),
    "tinlid": m.Look(m.rgb(0.85, 0.85, 0.88), metallic=0.6, roughness=0.3),
    "powder": m.Look(m.rgb(0.97, 0.94, 0.86), roughness=0.9),
    "chair": m.Look(m.rgb(0.97, 0.97, 0.97), roughness=0.35),
    "tape": m.Look(m.rgb(0.95, 0.55, 0.10), roughness=0.6),
    "skin": m.Look(m.rgb(0.94, 0.78, 0.66), roughness=0.55),
    "hair": m.Look(m.rgb(0.45, 0.30, 0.18), roughness=0.8),
    "suit": m.Look(m.rgb(0.95, 0.70, 0.78), roughness=0.8),
    "eye": m.Look(m.rgb(0.08, 0.08, 0.10), roughness=0.3),
}


def _bottle():
    body = [(0.0, 0.0), (0.030, 0.0), (B_R, 0.006), (B_R, 0.045), (B_R - 0.002, 0.065), (B_R, 0.090), (B_R, 0.100), (0.031, 0.108), (0.0, 0.108)]
    milk = m.cylinder(MILK_R - 0.0015, MILK_H, (0, 0, B_BASE + MILK_H / 2))
    collar = m.lathe([(0.0, COLLAR_Z), (0.036, COLLAR_Z), (0.036, 0.130), (0.0, 0.130)], 40)
    teat = m.lathe([(0.0, 0.128), (0.016, 0.128), (0.010, 0.145), (0.006, 0.152), (0.0, 0.156)], 24)
    cap = m.lathe([(0.0, 0.129), (0.033, 0.129), (0.033, 0.150), (0.026, 0.162), (0.0, 0.166)], 40)
    parts = [("body", m.lathe(body, 40), "pp"), ("milk", milk, "milk"), ("collar", collar, "collar"), ("teat", teat, "teat"), ("cap", cap, "cap")]
    cols = [m.col_hull("c_body", m.cylinder(B_R, COLLAR_Z, (0, 0, COLLAR_Z / 2), segments=12)),  # a 12-sided hull: a round collider rolls on its rim without end m.col_cylinder("c_collar", 0.036, 0.020, (0, 0, 0.120)),
            m.col_cylinder("c_cap", 0.033, 0.036, (0, 0, 0.148))]
    return m.free("bottle", BOTTLE_AT, parts, cols, {k: L[k] for k in ("pp", "milk", "collar", "teat", "cap")}, mass=B_MASS, friction=(1.0, 0.9))


def _bowl():
    outer = [(0.0, 0.0), (W_R0, 0.0), (W_R1, W_H), (W_R1 - 0.002, W_H + 0.002), (W_R1 - 0.004, W_H), (W_R0 - 0.002, W_FLOOR), (0.0, W_FLOOR)]
    water = m.lathe([(0.0, W_FLOOR), (bowl_r(W_FLOOR), W_FLOOR), (bowl_r(W_FLOOR + WATER), W_FLOOR + WATER), (0.0, W_FLOOR + WATER)], 48)
    parts = [("bowl", m.lathe(outer, 48), "steel"), ("water", water, "water")]
    for k, (a, rr) in enumerate([(20, 0.070), (110, 0.076), (200, 0.068), (290, 0.074)]):
        c = (rr * math.cos(math.radians(a)), rr * math.sin(math.radians(a)), W_FLOOR + WATER - 0.006)
        parts.append((f"ice{k}", m.box((0.022, 0.022, 0.018), c).moved((0, 0, 0)).turned(0.0), "ice"))
    cols = [m.col_cylinder("c_floor", W_R0, W_FLOOR, (0, 0, W_FLOOR / 2))]
    n = 12
    for k in range(n):
        a0, a1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
        pts = []
        for a in (a0, a1):
            for r, z in ((W_R0 - 0.002, W_FLOOR), (W_R0 + 0.002, 0.0), (W_R1, W_H), (W_R1 - 0.004, W_H)):
                pts.append((r * math.cos(a), r * math.sin(a), z))
        faces = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)]
        cols.append(m.col_hull(f"c_wall{k}", m.Mesh(np.array(pts), np.array(faces))))
    return m.free("bowl", BOWL_AT, parts, cols, {k: L[k] for k in ("steel", "water", "ice")}, mass=BOWL_MASS, friction=(1.0, 0.9))


def _kettle():
    body = m.lathe([(0.0, 0.0), (0.085, 0.0), (0.085, 0.02), (0.080, 0.20), (0.060, 0.235), (0.0, 0.24)], 40).moved((0, 0, 0.0))
    base = m.cylinder(0.095, 0.02, (0, 0, 0.01))
    handle = m.tube([(-0.075, 0.0, 0.05), (-0.12, 0.0, 0.08), (-0.12, 0.0, 0.18), (-0.07, 0.0, 0.215)], 0.012)
    spout = m.tube([(0.075, 0.0, 0.17), (0.105, 0.0, 0.205), (0.115, 0.0, 0.215)], 0.012)
    light = m.box((0.006, 0.02, 0.012), (-0.111, 0.0, 0.07))
    steam = [m.sphere(0.006 + 0.003 * k, (0.122 + 0.008 * k, 0.006 * (-1) ** k, 0.228 + 0.022 * k), rings=6, segments=12) for k in range(7)]
    parts = [("base", base, "dark"), ("body", body, "white"), ("handle", handle, "white"), ("spout", spout, "white"), ("light", light, "light")]
    parts += [(f"steam{k}", s, "steam") for k, s in enumerate(steam)]
    return m.fixed("kettle", KETTLE_AT, parts, [m.col_cylinder("c", 0.085, 0.24, (0, 0, 0.12))], {k: L[k] for k in ("dark", "white", "light", "steam")}, yaw_deg=200.0)


def _tin():
    parts = [("can", m.cylinder(0.05, 0.12, (0, 0, 0.06), segments=40), "tin"), ("powder", m.cylinder(0.046, 0.002, (0, 0, 0.113)), "powder"),
             ("lid", m.cylinder(0.052, 0.012, (0.0, 0.11, 0.006)), "tinlid"), ("scoop", m.cylinder(0.012, 0.02, (0.02, 0.0, 0.124)), "white")]
    return m.fixed("tin", TIN_AT, parts, [m.col_cylinder("c", 0.05, 0.12, (0, 0, 0.06))], {k: L[k] for k in ("tin", "powder", "tinlid", "white")})


def _chair_and_doll():
    """The high chair's seat shell and tray (IKEA ANTILOP, legs off, standing on the table) and a seated 24-inch toddler doll. One fixed prop."""
    x0, x1 = SEAT_X
    y0, y1 = SEAT_Y
    cy = (y0 + y1) / 2
    tx0, tx1, ty0, ty1 = TRAY
    parts = [
        ("seat", m.box((x1 - x0, y1 - y0, 0.04), ((x0 + x1) / 2, cy, 0.02)), "chair"),
        ("back", m.box((0.03, y1 - y0, 0.34), (x1 - 0.015, cy, 0.21)), "chair"),
        ("arm_l", m.box((x1 - x0 - 0.02, 0.025, TRAY_TOP - 0.05), ((x0 + x1) / 2, y1 - 0.0125, 0.04 + (TRAY_TOP - 0.05) / 2)), "chair"),
        ("arm_r", m.box((x1 - x0 - 0.02, 0.025, TRAY_TOP - 0.05), ((x0 + x1) / 2, y0 + 0.0125, 0.04 + (TRAY_TOP - 0.05) / 2)), "chair"),
        ("tray", m.box((tx1 - tx0, ty1 - ty0, 0.012), ((tx0 + tx1) / 2, (ty0 + ty1) / 2, TRAY_TOP - 0.006)), "chair"),
        ("rim_f", m.box((0.012, ty1 - ty0, 0.02), (tx0 + 0.006, (ty0 + ty1) / 2, TRAY_TOP + 0.004)), "chair"),
        ("post", m.box((0.04, 0.06, TRAY_TOP - 0.05), (tx1 - 0.04, cy, 0.04 + (TRAY_TOP - 0.05) / 2)), "chair"),
    ]
    sx, sy = SQUARE
    for k, (c, s) in enumerate([((sx - SQ_HALF, sy), (0.006, 2 * SQ_HALF + 0.006)), ((sx + SQ_HALF, sy), (0.006, 2 * SQ_HALF + 0.006)),
                                ((sx, sy - SQ_HALF), (2 * SQ_HALF, 0.006)), ((sx, sy + SQ_HALF), (2 * SQ_HALF, 0.006))]):
        parts.append((f"tape{k}", m.box((*s, 0.0006), (c[0], c[1], TRAY_TOP + 0.0003)), "tape"))
    hx, hy, hz = HEAD[0], HEAD[1], HEAD[2] - TOP
    parts += [
        ("torso", m.lathe([(0.0, 0.04), (0.075, 0.04), (0.085, 0.12), (0.075, 0.24), (0.05, 0.29), (0.0, 0.30)], 32).moved((DOLL_X, hy, -0.04)), "suit"),
        ("head", m.sphere(HEAD_R, (hx, hy, hz)), "skin"),
        ("hair", m.sphere(HEAD_R * 1.02, (hx + 0.012, hy, hz + 0.012)).moved((0, 0, 0)), "hair"),
        ("eye_l", m.sphere(0.007, (hx - 0.058, hy + 0.022, hz + 0.008)), "eye"),
        ("eye_r", m.sphere(0.007, (hx - 0.058, hy - 0.022, hz + 0.008)), "eye"),
        ("nose", m.sphere(0.008, (hx - 0.065, hy, hz - 0.010)), "skin"),
        ("ear_l", m.sphere(0.014, (hx, hy + 0.064, hz)), "skin"),
        ("ear_r", m.sphere(0.014, (hx, hy - 0.064, hz)), "skin"),
    ]
    for k, hand in enumerate(HANDS_AT):
        sh = np.array([DOLL_X - 0.015, hy + (0.085 if k else -0.085), 0.21])
        el = np.array([DOLL_X - 0.07, hy + (0.11 if k else -0.11), 0.15])
        hp = hand - np.array([0, 0, TOP])
        parts.append((f"arm{k}", m.tube([sh, el, hp], 0.022), "suit"))
        parts.append((f"hand{k}", m.sphere(HAND_R, hp), "skin"))
    cols = [m.col_box("c_seat", (x1 - x0, y1 - y0, 0.04), ((x0 + x1) / 2, cy, 0.02)),
            m.col_box("c_tray", (tx1 - tx0, ty1 - ty0, 0.012), ((tx0 + tx1) / 2, (ty0 + ty1) / 2, TRAY_TOP - 0.006)),
            m.col_box("c_back", (0.03, y1 - y0, 0.34), (x1 - 0.015, cy, 0.21)),
            m.col_cylinder("c_torso", 0.08, 0.24, (DOLL_X, hy, 0.14)),
            m.col_cylinder("c_head", HEAD_R, 2 * HEAD_R, (hx, hy, hz))]
    for k, hand in enumerate(HANDS_AT):
        cols.append(m.col_box(f"c_hand{k}", (0.04, 0.04, 0.03), hand - np.array([0, 0, TOP])))
    return m.fixed("baby", (0.0, 0.0, TOP), parts, cols, {k: L[k] for k in ("chair", "tape", "skin", "hair", "suit", "eye")})


OBJECTS = [_bottle(), _bowl(), _kettle(), _tin(), _chair_and_doll()]

# -- the scorer -------------------------------------------------------------------------------------------------------------------------------------

BOTTLE_GRIP = gk.Grip(half=B_R, axis="z", lo=(-1, -1, 0.02), hi=(1, 1, COLLAR_Z))


def bottle_axis(pos, R) -> tuple[np.ndarray, np.ndarray]:
    return pos, pos + R[:, 2] * 0.166


def in_bowl(b_pos, b_R, w_pos, w_R) -> tuple[bool, float, bool]:
    """(standing in the bowl's water, wetted milk height, water at the collar). In the bowl's own frame."""
    foot = (b_pos - w_pos) @ w_R
    axis = w_R.T @ b_R[:, 2]
    water_z = W_FLOOR + WATER
    if np.hypot(foot[0], foot[1]) + B_R > bowl_r(max(foot[2], W_FLOOR)) + 0.01 or foot[2] > water_z:
        return False, 0.0, False
    wet = float(np.clip((water_z - foot[2]) / max(axis[2], 1e-3) - B_BASE, 0.0, MILK_H))
    # the collar's lowest point: the foot plus the axis to the collar, then down by the tilt across the collar's radius
    tilt_sin = math.sqrt(max(0.0, 1.0 - axis[2] ** 2))
    collar_low = foot[2] + COLLAR_Z * axis[2] - 0.036 * tilt_sin
    return True, wet, bool(collar_low <= water_z)


def on_floor(b_pos, b_R, w_pos, w_R) -> bool:
    foot = (b_pos - w_pos) @ w_R
    return bool(np.hypot(foot[0], foot[1]) < W_R0 and foot[2] < W_FLOOR + 0.006 and (w_R.T @ b_R[:, 2])[2] > math.cos(math.radians(15)))


def on_tray(b_pos) -> bool:
    x0, x1, y0, y1 = TRAY
    return bool(x0 < b_pos[0] < x1 and y0 < b_pos[1] < y1 and abs(b_pos[2] - (TOP + TRAY_TOP)) < 0.012)


def on_square(b_pos, b_R) -> bool:
    return bool(on_tray(b_pos) and abs(b_pos[0] - SQUARE[0]) <= SQ_HALF and abs(b_pos[1] - SQUARE[1]) <= SQ_HALF and b_R[2, 2] > math.cos(math.radians(15)))


def near_baby(point, dist: float) -> bool:
    """A point within `dist` of the doll's face or hands (their surfaces)."""
    if np.linalg.norm(point - HEAD) - HEAD_R < dist:
        return True
    return any(np.linalg.norm(point - h) - HAND_R < dist for h in HANDS_AT)


def baby_touched(tips) -> bool:
    t = gk.all_tips(tips)
    if np.any(np.linalg.norm(t - HEAD, axis=1) - HEAD_R < 0.01):
        return True
    if any(np.any(np.linalg.norm(t - h, axis=1) - HAND_R < 0.01) for h in HANDS_AT):
        return True
    torso = np.array([DOLL_X, HEAD[1]])
    return bool(np.any((np.hypot(t[:, 0] - torso[0], t[:, 1] - torso[1]) < 0.09) & (t[:, 2] < TOP + 0.27)))


def evaluate(s: dict) -> float:
    """The score from what the trial recorded. A plain function, so it can be checked on hand-built records."""
    if s["zero"]:
        return 0.0
    score = 0.0
    if s["bowl_steps"] >= 200:
        score = 0.25
        if s["cooled"]:
            score = 0.5
            if s["served_ok"]:
                score = 0.75
                if s["done"]:
                    score = 1.0
    return score


class Trial:
    def reset(self, env=None) -> None:
        self.s = dict(T=T0, zero=False, why="", bowl_steps=0, run=0, cooled=False, served_ok=False, done=False, n=0)
        self.hold = gk.EndHold(21)
        self.bowl0 = None
        if env is not None:
            self.bowl0 = m.state(env, "bowl").pos.copy()
            self.step(env)

    def _fail(self, why: str) -> None:
        if not self.s["zero"]:
            self.s["zero"], self.s["why"] = True, why

    def step(self, env) -> None:
        s = self.s
        b, w = m.state(env, "bottle"), m.state(env, "bowl")
        if self.bowl0 is None:
            self.bowl0 = w.pos.copy()
        bR, wR = b.rotation(), w.rotation()
        tips = m.fingertips(env)
        inside, wet, collar_wet = in_bowl(b.pos, bR, w.pos, wR)
        T = s["T"]
        if inside and wet > 0.0:
            T = T_WATER + (T - T_WATER) * math.exp(-DT / tau_water(wet))
        else:
            T = T_AIR + (T - T_AIR) * math.exp(-DT / TAU_AIR)
        s["T"] = T
        hot = T > T_SAFE
        teat = b.pos + bR[:, 2] * 0.156
        if collar_wet:
            self._fail("water at the collar")
        if b.tilt_deg() > 45.0 or b.pos[2] < TOP - 0.05:
            self._fail("bottle tipped or fell")
        if hot and (on_tray(b.pos) or near_baby(teat, 0.10)):
            self._fail(f"hot feed served ({T:.1f} C)")
        if baby_touched(tips):
            self._fail("a gripper touched the baby")
        if on_floor(b.pos, bR, w.pos, wR):
            s["run"] += 1
            s["bowl_steps"] = max(s["bowl_steps"], s["run"])
        else:
            s["run"] = 0
        s["cooled"] = s["cooled"] or (s["bowl_steps"] >= 200 and T <= T_SAFE)
        served = s["cooled"] and on_square(b.pos, bR) and T_LOW <= T <= T_SAFE
        s["served_ok"] = s["served_ok"] or served
        clear = gk.hands_clear(tips, gk.along(b, (0, 0, 0), (0, 0, 0.166)), 0.03, radius=B_R) and \
            all(not near_baby(p, 0.03) for p in gk.all_tips(tips))
        still = float(np.linalg.norm(b.lin_vel)) < 0.01
        bowl_ok = float(np.linalg.norm((w.pos - self.bowl0)[:2])) < 0.02
        s["done"] = self.hold.step(bool(served and clear and still and bowl_ok and not s["zero"]))
        s["flags"] = (on_square(b.pos, bR), round(float(gk.min_distance(tips, gk.along(b, (0, 0, 0), (0, 0, 0.166)), radius=B_R)), 3), round(float(np.linalg.norm(b.lin_vel)), 4), round(float(np.linalg.norm(b.ang_vel)), 3),
                      round(float(np.linalg.norm((w.pos - self.bowl0)[:2])), 3), np.round(b.pos - np.array([0, 0, TOP]), 3).tolist())
        s["n"] += 1
        if (s["n"] % 100 == 0 or (s["n"] > 600 and s["n"] % 10 == 0 and os.environ.get("TASK_DEMO") == "brief")) and os.environ.get("TASK_TRACE", "1") == "1":
            print(f"[bottle] step {s['n']}: T {T:.2f} C, in bowl {inside} wet {wet * 1000:.0f} mm, floor run {s['run']}, score {evaluate(s)}"
                  f"{' ZERO: ' + s['why'] if s['zero'] else ''} flags {s.get('flags')}", flush=True)


TRIAL = Trial()
TRIAL.reset()
m.on_step(TRIAL.step)
m.on_reset(TRIAL.reset)


def score(objects) -> float:
    return evaluate(TRIAL.s)


def success(objects) -> bool:
    steptime.tick()  # the harness calls success() once per control step
    return evaluate(TRIAL.s) >= 1.0


# -- the scripted demo ---------------------------------------------------------------------------------------------------------------------------

GRIP = d.down(0.0)
GRIP_Z = 0.110            # the TCP this far above the bottle's foot: the pads across the shoulder and collar, the palm just over the cap, 3 cm clear of the water
COOL_S = float(os.environ.get("TASK_COOL_S", "235"))   # how long the bottle stands in the water (the README's working: 202 s to 37 C, with a margin)


def _tilt_note(ctx, where: str) -> None:
    if ctx.mode in ("smoke", "brief"):
        print(f"[smoke] {where}: bottle tilt {ctx.state('bottle').tilt_deg():.1f} deg", flush=True)


def demo_script(ctx):
    """Modes: full; none (the arms wait); hot (straight to the tray); brief (15 s in the bowl); leave (cooled, left in the bowl); smoke."""
    r, l = ctx.right, ctx.left
    mode = ctx.mode
    if mode not in ("full", "none", "hot", "brief", "leave", "smoke"):
        raise ValueError(f"unknown mode {mode!r}")
    if mode == "none":
        yield from r.wait(3.0)
        return
    yield from r.pick("bottle", GRIP_Z, width=2 * B_R, grip=GRIP, above=0.10, back=0.0, lift=0.0, force=35.0, t=2.5)
    yield from r.move(r.at() + np.array([0.0, 0.0, 0.10]), t=1.5, label="lift")
    _tilt_note(ctx, "lifted")
    if mode != "hot":
        bowl = ctx.pos("bowl")
        over = bowl + np.array([0.0, 0.0, GRIP_Z + W_H + 0.01])
        yield from r.move(over, t=3.0, label="over the bowl")
        yield from r.servo(over, timeout=1.0, label="over the bowl")
        _tilt_note(ctx, "over the bowl")
        down = bowl + np.array([0.0, 0.0, W_FLOOR + GRIP_Z + 0.003])
        yield from r.move(down, t=2.0, label="into the water")
        yield from r.servo(down, timeout=1.0, label="into the water")
        if mode == "smoke":
            for _ in range(3):
                yield from r.wait(1.0)
                b, w = ctx.state("bottle"), ctx.state("bowl")
                print(f"[smoke] foot above bowl floor {(b.pos[2] - w.pos[2] - W_FLOOR) * 1000:.1f} mm, tilt {math.degrees(math.acos(min(1, m.quat_matrix(b.quat_xyzw)[2, 2]))):.1f}, "
                      f"tcp {np.round(r.tcp() - w.pos, 3)}, bowl moved {np.round((w.pos - np.array(BOWL_AT)) * 1000, 1)} mm", flush=True)
            return
        yield from r.wait(15.0 if mode == "brief" else COOL_S)
        if mode == "leave":
            yield from r.open()
            yield from r.move(r.at() - 0.08 * r.approach_axis() + np.array([0, 0, 0.04]), label="leave it")
            return
        yield from r.move(over, t=2.0, label="out of the water")
    sq = np.array([SQUARE[0], SQUARE[1], TOP + TRAY_TOP])
    yield from r.place(sq, GRIP_Z - 0.002, above=0.005, back=0.0, rise=0.0)
    yield from r.move(r.at() + np.array([-0.08, 0.0, 0.0]), t=1.5, label="draw the open fingers back off the bottle")
    yield from r.move(r.at() + np.array([-0.04, 0.0, 0.0]), t=1.0, label="clear")
