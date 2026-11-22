"""A loaded E-Z curl bar lies level in two padded notched cradles. Each sleeve carries a 1.25 lb plate held by a lever-lock collar
(lever down = clamped). Two 2.5 lb plates stand in slot trays in front of the bar, between the arms. The robot is asked to add one of them to
each end of the bar.

The safe way, the weight-room rule: plates go inside a locked collar. Flip the collar's lever up, slide the collar off the
sleeve end and set it in its rest, thread the new plate on against the first one, slide the collar back against it and press
its lever down. A plate hung on the bare sleeve outside the collar slides off as soon as the bar tips.

Positions are in metres in the station frame (x forward, y left); heights are above the table top unless `TOP` is added. The
bar's, plates' and collars' main sizes follow the makers' listings (1-inch sleeves, standard 1-inch-hole plates); the plates'
diameters and thicknesses, the collar's size and lever, and the masses of the collar and lever are engineering estimates.
In the simulation the collar's clamp is its lever: lever folded flat = clamped. The clamping force itself is not simulated;
a clamped collar that is dragged along the sleeve counts as a failure.
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import gatekit as gk  # noqa: E402
import mechkit as m  # noqa: E402
import poseplay  # noqa: E402,F401
import steptime  # noqa: E402
from poseplay import Pose  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402
from yam_arena.urdf import axis_angle_matrix  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
INSTRUCTION = "Add one of the 2.5 lb plates to each end of the curl bar, then let go and move both hands away."
SUCCESS_HOLD_S = 0.0
SCORE_OVER = "end"
SIDES = (("left", 1.0), ("right", -1.0))

# -- the bar: CAP standard 1-inch E-Z curl bar, 47 in (1.19 m), plain 1-inch sleeves; fixed in its cradles ------------------------
BX, BZ = 0.25, 0.13  # the bar's axis: along y, at x 25 cm, 13 cm above the table
SLEEVE_R = 0.0127  # 1 inch across
SHOULDER_Y = 0.397  # the inner sleeve collar's outer face (from the bar's middle)
END_Y = 0.597  # the sleeve's end
CHAMFER = 0.004  # the sleeve end's chamfer
SHOULDER_R, SHOULDER_W = 0.019, 0.014
CRADLE_Y = 0.33

# -- plates: standard cast-iron 1-inch-hole plates (hole 1-1/8 in, 28.6 mm) ------------------------------------------------------
HOLE_R = 0.0143
SMALL = dict(r=0.070, t=0.016, mass=0.567)  # 1.25 lb
BIG = dict(r=0.0825, t=0.020, mass=1.134)  # 2.5 lb
SMALL_Y = SHOULDER_Y + SMALL["t"] / 2

# -- the lever-lock collar: nylon body, 58 mm across, 40 mm long, a flip lever on top -------------------------------------------
COL_R, COL_L, BORE_R, COL_MASS = 0.029, 0.040, 0.0140, 0.12
COL_Y0 = SHOULDER_Y + SMALL["t"] + COL_L / 2  # its middle, at the start
LEV_L, LEV_W, LEV_T, FIN_H = 0.034, 0.018, 0.004, 0.014
LEV_OPEN = 45.0  # the lever's stop, degrees up: open (the clamp released) from about 30
LOCKED_DEG = 8.0  # lever within this of flat: locked, for the score
GRIP_DEG = 30.0  # lever below this: the jaw still grips, so the collar must not be slid (the drag rule)

# -- the plate tray (two slots) and the collar rest -------------------------------------------------------------------------
TRAY_X, TRAY_Y, FLOOR = 0.42, 0.15, 0.012
SLOT = BIG["t"] + 0.006
REST_X, REST_Y, REST_TOP, REST_GROOVE = 0.12, 0.50, 0.050, 0.040
REST_Z = REST_TOP + math.sqrt(COL_R ** 2 - (REST_GROOVE / 2) ** 2)  # the collar's axis, sitting in the groove

LOOKS = {
    "chrome": m.Look(m.rgb(0.80, 0.81, 0.83), metallic=1.0, roughness=0.18),
    "iron": m.Look(m.rgb(0.10, 0.10, 0.11), metallic=0.6, roughness=0.55),
    "nylon": m.Look(m.rgb(0.05, 0.05, 0.05), roughness=0.5),
    "lever": m.Look(m.rgb(0.75, 0.08, 0.06), roughness=0.45),
    "wood": m.Look(m.rgb(0.70, 0.52, 0.32), roughness=0.8),
    "felt": m.Look(m.rgb(0.15, 0.17, 0.20), roughness=1.0),
    "ply": m.Look(m.rgb(0.78, 0.66, 0.46), roughness=0.75),
    "white": m.Look(m.rgb(0.92, 0.92, 0.92), roughness=0.6),
}


def _col_box_ry(name: str, size, center, deg: float) -> str:
    """A box collider turned `deg` about y."""
    return m.col_box(name, size, center).replace(
        'uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
        f'float xformOp:rotateY = {deg:.4f}\n        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateY", "xformOp:scale"]')


def ring_cols(prefix: str, r_in: float, r_out: float, y0: float, y1: float, n: int, cx: float = 0.0, cz: float = 0.0) -> list[str]:
    """Boxes round the y axis between two radii: a ring threaded on a bar along y."""
    r_mid, thick = (r_in + r_out) / 2, r_out - r_in
    w = 2 * r_out * math.tan(math.pi / n) * 1.02
    out = []
    for i in range(n):
        a = 2 * math.pi * (i + 0.5) / n
        c = (cx + r_mid * math.cos(a), (y0 + y1) / 2, cz + r_mid * math.sin(a))
        out.append(_col_box_ry(f"{prefix}{i}", (thick, y1 - y0, w), c, -math.degrees(a)))
    return out


def _ring_y(r_in: float, r_out: float, t: float, lip: float = 0.0, center=(0.0, 0.0, 0.0)) -> m.Mesh:
    """A flat ring about y, `t` thick; `lip` raises a rim on both faces (a cast plate's edge)."""
    h = t / 2
    if lip:
        prof = [(r_in, -h + 0.002), (r_in + 0.010, -h + 0.002), (r_in + 0.012, -h), (r_out - 0.008, -h), (r_out - 0.006, -h - lip),
                (r_out, -h - lip), (r_out, h + lip), (r_out - 0.006, h + lip), (r_out - 0.008, h), (r_in + 0.012, h), (r_in + 0.010, h - 0.002),
                (r_in, h - 0.002), (r_in, -h + 0.002)]
    else:
        prof = [(r_in, -h), (r_out, -h), (r_out, h), (r_in, h), (r_in, -h)]
    return m.lathe(prof, 48).aimed("+Y").moved(center)


def _bar():
    parts, cols = [], []
    for k, s in enumerate((1.0, -1.0)):
        y_in, y_out = s * SHOULDER_Y, s * (END_Y - CHAMFER)
        mid = (y_in + y_out) / 2
        parts.append((f"sleeve{k}", m.cylinder(SLEEVE_R, abs(y_out - y_in), (0.0, mid, 0.0), axis="y", segments=32), "chrome"))
        cols.append(m.col_hull(f"sleeve{k}_c", m.cylinder(SLEEVE_R, abs(y_out - y_in), (0.0, mid, 0.0), axis="y", segments=24)))
        cham = m.lathe([(0.0, 0.0), (SLEEVE_R, 0.0), (SLEEVE_R - 0.0035, CHAMFER), (0.0, CHAMFER)], 24).aimed("+Y" if s > 0 else "-Y").moved((0.0, y_out, 0.0))
        parts.append((f"end{k}", cham, "chrome"))
        cols.append(m.col_hull(f"end{k}_c", cham))
        sh = m.cylinder(SHOULDER_R, SHOULDER_W, (0.0, s * (SHOULDER_Y - SHOULDER_W / 2), 0.0), axis="y")
        parts.append((f"shoulder{k}", sh, "chrome"))
        cols.append(m.col_hull(f"shoulder{k}_c", sh))
    # the handle: 1-inch bar, straight inside each shoulder, then the E-Z bends in the middle (seen only; nothing reaches it)
    pts = [(0.0, SHOULDER_Y - SHOULDER_W, 0.0), (0.0, 0.26, 0.0), (0.022, 0.20, 0.0), (0.0, 0.13, 0.0), (0.022, 0.06, 0.0)]
    path = pts + [(x, -y, z) for x, y, z in reversed(pts)]
    parts.append(("handle", m.tube(path, 0.0125, 16), "chrome"))
    # the two padded notched cradles under the straight handle sections, fixed to the table
    for k, s in enumerate((1.0, -1.0)):
        y = s * CRADLE_Y
        for side in (-1.0, 1.0):
            blk = m.box((0.035, 0.09, BZ - 0.003), (side * 0.028, y, -(BZ + 0.003) / 2 - 0.003))
            parts.append((f"cradle{k}{int(side > 0)}", blk, "wood"))
        parts.append((f"felt{k}", m.box((0.022, 0.09, 0.003), (0.0, y, -SLEEVE_R - 0.0015)), "felt"))
        parts.append((f"block{k}", m.box((0.095, 0.09, BZ - SLEEVE_R - 0.003), (0.0, y, -BZ + (BZ - SLEEVE_R - 0.003) / 2)), "wood"))
    return m.fixed("bar", (BX, 0.0, TOP + BZ), parts, cols, LOOKS)


def _plate(name: str, spec: dict, pos, color: str = "iron"):
    r, t = spec["r"], spec["t"]
    vis = [("body", _ring_y(HOLE_R, r, t, lip=0.0015), color),
           ("mark", m.box((0.03, 0.0005, 0.010), (0.0, t / 2 + 0.0018, r * 0.62)), "white"),
           ("mark2", m.box((0.03, 0.0005, 0.010), (0.0, -t / 2 - 0.0018, r * 0.62)), "white")]
    hub = HOLE_R + 0.016
    cols = ring_cols(f"{name}_hub", HOLE_R, hub, -t / 2, t / 2, 16) + ring_cols(f"{name}_disc", hub - 0.002, r, -t / 2, t / 2, 24)
    return m.free(name, pos, vis, cols, LOOKS, mass=spec["mass"], friction=(0.6, 0.5))


def _collar(name: str, s: float, pos):
    """A lever-lock collar about the y axis. `s` = +1 for the left sleeve: its lever's hinge is on the plate side (local -y),
    lying outward along local +y; the right collar is the same part turned 180 degrees."""
    body = m.lathe([(BORE_R, -COL_L / 2), (COL_R, -COL_L / 2), (COL_R, COL_L / 2), (BORE_R, COL_L / 2), (BORE_R, -COL_L / 2)], 40).aimed("+Y")
    pad = m.box((LEV_W + 0.006, COL_L - 0.004, 0.003), (0.0, 0.0, COL_R - 0.001))  # the flat the lever folds onto
    hz = COL_R + 0.002 + LEV_T / 2
    hy = -COL_L / 2 + 0.003
    lev = m.box((LEV_W, LEV_L, LEV_T), (0.0, hy + LEV_L / 2, hz))
    fin = m.box((LEV_W, 0.004, FIN_H), (0.0, hy + LEV_L - 0.002, hz + LEV_T / 2 + FIN_H / 2))
    lever = m.hinge("lever", "X", (0.0, hy, hz), 0.0, LEV_OPEN, [("lever", lev, "lever"), ("fin", fin, "lever")],
                    [m.col_box("lever_c", (LEV_W, LEV_L, LEV_T), (0.0, hy + LEV_L / 2, hz)),
                     m.col_box("fin_c", (LEV_W, 0.004, FIN_H), (0.0, hy + LEV_L - 0.002, hz + LEV_T / 2 + FIN_H / 2))],
                    mass=0.01, friction=0.01, damping=0.01)
    cols = ring_cols(f"{name}_r", BORE_R, COL_R, -COL_L / 2, COL_L / 2, 20)
    cols.append(m.col_box(f"{name}_pad", (LEV_W + 0.006, COL_L - 0.004, 0.003), (0.0, 0.0, COL_R - 0.001)))
    return m.mechanism(name, pos, [lever], LOOKS, frame_parts=[("body", body, "nylon"), ("pad", pad, "nylon")], frame_colliders=cols,
                       free_mass=COL_MASS, yaw_deg=0.0 if s > 0 else 180.0, friction=(0.5, 0.4), frame_inertia=(5e-5, 5e-5, 5e-5))


def _tray(s: float):
    parts, cols = [], []
    L, wall_t, wall_h = 0.20, 0.012, 0.040
    parts.append(("floor", m.box((L, SLOT + 2 * wall_t, FLOOR), (0.0, 0.0, FLOOR / 2)), "ply"))
    cols.append(m.col_box("floor_c", (L, SLOT + 2 * wall_t, FLOOR), (0.0, 0.0, FLOOR / 2)))
    for k, sy in enumerate((-1.0, 1.0)):
        c = (0.0, sy * (SLOT / 2 + wall_t / 2), FLOOR + wall_h / 2)
        parts.append((f"wall{k}", m.box((L, wall_t, wall_h), c), "ply"))
        cols.append(m.col_box(f"wall{k}_c", (L, wall_t, wall_h), c))
        e = (sy * (BIG["r"] + 0.004 + 0.006), 0.0, FLOOR + 0.0125)
        parts.append((f"stop{k}", m.box((0.012, SLOT, 0.025), e), "ply"))
        cols.append(m.col_box(f"stop{k}_c", (0.012, SLOT, 0.025), e))
    return m.fixed(f"tray_{'l' if s > 0 else 'r'}", (TRAY_X, s * TRAY_Y, TOP), parts, cols, LOOKS)


def _rest(s: float):
    w, L = 0.080, 0.060
    parts, cols = [], []
    for k, sx in enumerate((-1.0, 1.0)):
        bw = (w - REST_GROOVE) / 2
        c = (sx * (REST_GROOVE / 2 + bw / 2), 0.0, REST_TOP / 2)
        parts.append((f"side{k}", m.box((bw, L, REST_TOP), c), "wood"))
        cols.append(m.col_box(f"side{k}_c", (bw, L, REST_TOP), c))
    parts.append(("base", m.box((REST_GROOVE, L, REST_TOP - 0.022), (0.0, 0.0, (REST_TOP - 0.022) / 2)), "wood"))
    cols.append(m.col_box("base_c", (REST_GROOVE, L, REST_TOP - 0.022), (0.0, 0.0, (REST_TOP - 0.022) / 2)))
    return m.fixed(f"rest_{'l' if s > 0 else 'r'}", (REST_X, s * REST_Y, TOP), parts, cols, LOOKS)


def _hang_z(hole_r: float) -> float:
    return TOP + BZ - (hole_r - SLEEVE_R)


OBJECTS = [_bar()]
for _side, _s in SIDES:
    OBJECTS += [_plate(f"small_{_side}", SMALL, (BX, _s * SMALL_Y, _hang_z(HOLE_R))),
                _collar(f"collar_{_side}", _s, (BX, _s * COL_Y0, _hang_z(BORE_R))),
                _plate(f"big_{_side}", BIG, (TRAY_X, _s * TRAY_Y, TOP + FLOOR + BIG["r"] + 0.001)),
                _tray(_s), _rest(_s)]

# -- reading the state ------------------------------------------------------------------------------------------------------


def on_sleeve(pos, s: float, tol: float = 0.004) -> bool:
    """A ring's centre on sleeve `s`: within `tol` of the axis line (it hangs by its hole) and between the shoulder and the end."""
    r = math.hypot(pos[0] - BX, pos[2] - (TOP + BZ))
    return r <= tol and SHOULDER_Y - 0.002 <= s * pos[1] <= END_Y


def y_half(st, r: float, half_t: float) -> float:
    """Half a ring's extent along y (the sleeve), from its rotation: a ring of radius r, half_t thick about its own y axis."""
    R = st.rotation()
    return float(abs(R[1, 1]) * half_t + r * math.hypot(R[1, 0], R[1, 2]))


def pad_points(tips: np.ndarray, n: int = 7) -> np.ndarray:
    t = np.asarray(tips, float).reshape(-1, 2, 3)
    w = np.linspace(0.0, 1.0, n)[None, :, None]
    return (t[:, :1, :] * (1 - w) + t[:, 1:, :] * w).reshape(-1, 3)


def ring_gap(points: np.ndarray, center, r: float, half_t: float) -> float:
    """Smallest distance from points to a disc of radius r about y (an axis-aligned ring's outside)."""
    d = np.asarray(points, float) - np.asarray(center, float)
    rad = np.hypot(d[:, 0], d[:, 2])
    dr = np.maximum(rad - r, 0.0)
    dy = np.maximum(np.abs(d[:, 1]) - half_t, 0.0)
    return float(np.min(np.hypot(dr, dy)))


def bar_gap(points: np.ndarray) -> float:
    """Distance from points to the sleeves (and their shoulders)."""
    d = np.asarray(points, float)
    rad = np.hypot(d[:, 0] - BX, d[:, 2] - (TOP + BZ))
    ay = np.abs(d[:, 1])
    dy = np.maximum(np.maximum(SHOULDER_Y - SHOULDER_W - ay, ay - END_Y), 0.0)
    return float(np.min(np.hypot(np.maximum(rad - SHOULDER_R, 0.0), dy)))


# -- the scorer -------------------------------------------------------------------------------------------------------------

NEXT_TO = 0.010  # the new plate within 1 cm of the 1.25 lb plate
COLLAR_GAP = 0.005  # the collar within 5 mm of the outer plate
CLEAR = 0.04
STILL = 0.002
HOLD_STEPS = 20
DRAG = 0.005  # a clamped collar moved this far along its sleeve: forced
# a part's centre below these heights (m above the table) has fallen: lower than standing in its tray (2.5 lb), hanging on its sleeve
# (1.25 lb) or sitting in its rest (collar); a plate on its edge on the table, or a collar on the table or a tray floor, is below
LOWEST = {"big": FLOOR + BIG["r"] - 0.005, "small": SMALL["r"] + 0.03, "collar": REST_Z - 0.011}


class Trial:
    def __init__(self):
        self.reset()

    def reset(self, env=None) -> None:
        self.n = 0
        self.fell = gk.Latch("a plate or collar fell off the bar or out of reach")
        self.dragged = gk.Latch("a clamped collar was dragged along its sleeve")
        self.clamp_y = {}
        self.hist = []
        self.done_steps = 0
        self.score, self.full = 0.0, False
        self.state = {}

    def failed(self) -> bool:
        return bool(self.fell or self.dragged)

    def end(self, side: str, s: float, small, bigs: list, collar, lever_deg: float) -> dict:
        """One end's state: is a 2.5 lb plate (either one: they are identical) inside a clamped collar, and is anything outside or
        unclamped. `bigs` lists both 2.5 lb plates, this side's first."""
        on_this = [b for b in bigs if on_sleeve(b.pos, s)]
        big = min(on_this, key=lambda b: s * b.pos[1]) if on_this else bigs[0]
        sp, bp, cp = small.pos, big.pos, collar.pos
        small_on, big_on, col_on = on_sleeve(sp, s), on_sleeve(bp, s), on_sleeve(cp, s)
        clamped = col_on and lever_deg <= LOCKED_DEG
        sy, by, cy = s * sp[1], s * bp[1], s * cp[1]
        # faces along the sleeve, including a ring's lean on the sleeve (a hanging plate leans a few degrees in its 1.6 mm play):
        # the gaps are between the nearest points, as a tape would read them
        hs, hb, hc = y_half(small, SMALL["r"], SMALL["t"] / 2), y_half(big, BIG["r"], BIG["t"] / 2), y_half(collar, COL_R, COL_L / 2)
        small_out = sy + hs
        gap_plates = (by - hb) - small_out
        gap_collar = (cy - hc) - (by + hb)
        inside = big_on and col_on and by > sy and gap_plates <= NEXT_TO and by < cy
        snug = clamped and inside and gap_collar <= COLLAR_GAP
        # a plate on the sleeve with no clamped collar outside it
        loose = any(on and (not clamped or y >= cy) for on, y in [(small_on, sy)] + [(True, s * b.pos[1]) for b in on_this])
        return dict(done=bool(snug and small_on), loose=bool(loose), small_on=small_on, big_on=big_on, col_on=col_on, clamped=clamped,
                    inside=bool(inside), lever=lever_deg, gap_plates=gap_plates, gap_collar=gap_collar)

    def observe(self, states: dict, tips: np.ndarray) -> None:
        self.n += 1
        pads = pad_points(tips)
        ends = {}
        for side, s in SIDES:
            col = states[f"collar_{side}"]
            lever = col.deg("lever")
            other = "right" if side == "left" else "left"
            e = self.end(side, s, states[f"small_{side}"], [states[f"big_{side}"], states[f"big_{other}"]], col, lever)
            ends[side] = e
            if self.n > 5:
                if e["col_on"] and lever <= GRIP_DEG:
                    y0 = self.clamp_y.setdefault(side, s * col.pos[1])
                    self.dragged.trip(abs(s * col.pos[1] - y0) > DRAG, self.n)
                else:
                    self.clamp_y.pop(side, None)
        if self.n > 5:
            for name, st in states.items():
                if name.startswith(("small", "big", "collar")):
                    self.fell.trip(st.pos[2] < TOP + LOWEST[name.split("_")[0]] or st.pos[0] > 0.6 or abs(st.pos[1]) > 0.75, self.n)
        moving = [states[k].pos for k in sorted(states) if k.startswith(("small", "big", "collar"))]
        self.hist = (self.hist + [np.concatenate(moving)])[-(HOLD_STEPS + 1):]
        still = len(self.hist) > HOLD_STEPS and float(np.max(np.ptp(np.asarray(self.hist), axis=0))) < STILL  # every sample of the last 1 s
        gaps = [bar_gap(pads)]
        for side, _ in SIDES:
            for kind, spec in (("small", SMALL), ("big", BIG)):
                gaps.append(ring_gap(pads, states[f"{kind}_{side}"].pos, spec["r"], spec["t"] / 2))
            gaps.append(ring_gap(pads, states[f"collar_{side}"].pos, COL_R + LEV_L, COL_L / 2))
        clear = min(gaps) >= CLEAR
        any_loose = any(e["loose"] for e in ends.values())
        done = [e["done"] for e in ends.values()]
        ok = all(done) and clear and still
        self.done_steps = self.done_steps + 1 if ok else 0
        self.full = (not self.failed()) and self.done_steps > HOLD_STEPS
        if self.failed() or any_loose:
            self.score = 0.0
        elif self.full:
            self.score = 1.0
        else:
            self.score = 0.5 * min(sum(done), 1) if sum(done) == 1 else (0.5 if all(done) else 0.0)
        self.state = dict(ends=ends, clear=clear, clear_m=min(gaps), still=still)


TRIAL = Trial()
ENV = [None]
NAMES = [f"{k}_{side}" for side, _ in SIDES for k in ("small", "big", "collar")]


@m.on_reset
def _reset(env) -> None:
    ENV[0] = env
    TRIAL.reset(env)


@m.on_step
def _step(env) -> None:
    ENV[0] = env
    states = {n: m.state(env, n) for n in NAMES}
    TRIAL.observe(states, m.fingertips(env))
    if os.environ.get("TASK_TRACE") and TRIAL.n % 10 == 0:
        st = TRIAL.state
        parts = []
        for side, _ in SIDES:
            e = st["ends"][side]
            b, c = states[f"big_{side}"].pos, states[f"collar_{side}"].pos
            parts.append(f"{side}: big {np.round(b - [0, 0, TOP], 3)} collar {np.round(c - [0, 0, TOP], 3)} lever {e['lever']:.1f} tilt {states[f'collar_{side}'].tilt_deg():.1f} "
                         f"on s/b/c {e['small_on']}/{e['big_on']}/{e['col_on']} clamped {e['clamped']} inside {e['inside']} "
                         f"gaps {e['gap_plates'] * 1000:.1f}/{e['gap_collar'] * 1000:.1f} mm done {e['done']} loose {e['loose']}")
        print(f"[bar] step {TRIAL.n}: " + " | ".join(parts) + f" | clear {st['clear_m'] * 1000:.0f} mm still {st['still']} "
              f"fell {bool(TRIAL.fell)} dragged {bool(TRIAL.dragged)} score {TRIAL.score}", flush=True)


def success(objects) -> bool:
    steptime.tick()
    return TRIAL.full


def score(objects) -> float:
    return TRIAL.score


# -- the scripted demo ---------------------------------------------------------------------------------------------------

def _tilted(theta_deg: float, phi_deg: float, yaw: float | None = None) -> np.ndarray:
    """Fingers tilted `theta_deg` from straight down toward the direction `phi_deg` from +x; they close across `yaw` (default: across phi)."""
    phi = math.radians(phi_deg)
    g = grip_rotation(phi if yaw is None else yaw)
    return axis_angle_matrix(np.array([math.sin(phi), -math.cos(phi), 0.0]), math.radians(theta_deg)) @ g


PLATE_ROT = _tilted(float(os.environ.get("PLATE_TILT", "40")), 0.0, yaw=0.0)  # leaning forward, closing across the plate's thickness
COLLAR_TILT = float(os.environ.get("COLLAR_TILT", "20"))
OPEN_F = FINGER_OPEN
SHUT = 0.0
OUT_X = 0.43  # the plate's carry line in front of the bar: clear of the 1.25 lb plate's rim (x 32 cm)
AWAY = tuple(float(v) for v in os.environ.get("AWAY_POS", "0.40,0.16,0.24").split(","))


def _w(p) -> np.ndarray:
    return np.array([float(p[0]), float(p[1]), float(p[2]) + TOP])


def _tcp_for_tips(p, rot) -> np.ndarray:
    """The TCP that puts the closed fingertips' end at station point `p` (TCP 13 mm up the fingers)."""
    z = np.asarray(rot, float) @ np.array([0.0, 0.0, 1.0])
    return np.asarray(p, float) - 0.013 * z


def _squeeze(width: float, newtons: float) -> float:
    return -max(0.0, width / 2 - newtons / 5000.0)


PLATE_SQ = _squeeze(BIG["t"] + 0.003, 38.0)
COLLAR_SQ = _squeeze(2 * COL_R, 30.0)


class Arm:
    """One arm's plan for its own end. Positions are written for the left end and mirrored in y by `s`."""

    def __init__(self, side: str, s: float):
        self.side, self.s = side, s
        self.cmd = None  # the last commanded TCP (station frame, TOP added)
        global COLLAR_ROT, PUSH_ROT
        COLLAR_ROT = _tilted(COLLAR_TILT, 0.0, yaw=s * math.pi / 2)  # closing across the collar, along x
        PUSH_ROT = grip_rotation(s * math.pi / 2)

    def P(self, x, y, z) -> np.ndarray:
        return _w((x, self.s * y, z))

    def state(self, name: str):
        return None if ENV[0] is None else m.state(ENV[0], f"{name}_{self.side}")

    def pose(self, t, pos, finger, label, rot):
        """A waypoint whose target is fixed (a point) or found when it starts (a callable of nothing)."""
        def f(plan, pos=pos):
            p = np.asarray(pos() if callable(pos) else pos, float)
            self.cmd = p
            return tuple(p)
        return Pose(t, f, finger, f"{self.side}: {label}", rot=rot)

    def follow(self, t, name: str, grip_off, want, finger, label, rot, gain: float = 1.0):
        """Move so that part `name`'s centre goes to `want` (station point, TOP added): from the last command, by where the part is now."""
        def target():
            st = self.state(name)
            if st is None or self.cmd is None:
                return np.asarray(want, float) + grip_off
            return self.cmd + gain * (np.asarray(want, float) - st.pos)
        return self.pose(t, target, finger, label, rot)

    def at_part(self, name: str, off, nominal) -> callable:
        """A TCP at part `name`'s centre plus `off`, where the part is now."""
        def target():
            st = self.state(name)
            c = np.asarray(nominal, float) if st is None else st.pos
            return c + np.asarray(off, float)
        return target

    # -- the collar's lever, pushed with the closed fingertips ------------------------------------------------------------------
    def lever_tip(self, collar_y: float, deg: float, r: float, lead: float = 0.004) -> np.ndarray:
        """A point on the lever's arc, `r` from its hinge, at `deg` up, `lead` metres ahead along the push."""
        hy = collar_y - COL_L / 2 + 0.003
        hz = BZ - (BORE_R - SLEEVE_R) + COL_R + 0.002 + LEV_T / 2
        a = math.radians(deg)
        p = np.array([BX, hy + r * math.cos(a), hz + r * math.sin(a)])
        tang = np.array([0.0, -math.sin(a), math.cos(a)])
        return p + lead * tang

    def open_lever(self) -> list:
        cy = COL_Y0
        r_fin = LEV_L  # the fin's outer face, at the lever's free end
        hz_fin = 0.008
        out = []
        start = self.lever_tip(cy, 0.0, r_fin + 0.012, 0.0) + np.array([0.0, 0.0, hz_fin])
        out.append(self.pose(3.0, self.P(*_tcp_for_tips(start + [0, 0, 0.03], PUSH_ROT)), SHUT, "over the collar's lever, fingers shut", PUSH_ROT))
        out.append(self.pose(1.5, self.P(*_tcp_for_tips(start, PUSH_ROT)), SHUT, "fingertips beside the lever's fin", PUSH_ROT))
        for deg, t in ((10.0, 1.0), (30.0, 1.0), (50.0, 1.0), (LEV_OPEN + 6.0, 1.0)):
            a = math.radians(deg)
            hy = cy - COL_L / 2 + 0.003
            hz = BZ - (BORE_R - SLEEVE_R) + COL_R + 0.002 + LEV_T / 2
            # the fin's outer face turns about the hinge: push on it from outside, a little ahead
            fin = np.array([BX, hy + r_fin * math.cos(a) - hz_fin * math.sin(a), hz + r_fin * math.sin(a) + hz_fin * math.cos(a)])
            ahead = np.array([0.0, -math.cos(a), -math.sin(a)]) * 0.004  # into the fin
            out.append(self.pose(t, self.P(*_tcp_for_tips(fin + ahead, PUSH_ROT)), SHUT, f"push the lever up, {deg:.0f} deg", PUSH_ROT))
        out.append(self.pose(1.0, self.P(*(_tcp_for_tips(fin, PUSH_ROT) + [0, 0.0, 0.02])), SHUT, "up off the lever", PUSH_ROT))
        return out

    def close_lever(self, cy: float) -> list:
        hy = cy - COL_L / 2 + 0.003
        hz = BZ - (BORE_R - SLEEVE_R) + COL_R + 0.002 + LEV_T / 2
        r = LEV_L - 0.005
        out = []
        for deg, t, lab in ((LEV_OPEN + 4.0, 2.5, "fingertips over the raised lever"), (LEV_OPEN - 2.0, 1.0, "on the lever"), (40.0, 1.0, "press it down"),
                            (15.0, 1.0, "press it down"), (-22.0, 1.5, "press it flat")):
            a = math.radians(deg)
            p = np.array([BX, hy + r * math.cos(a), hz + r * math.sin(a) + LEV_T / 2 + 0.002])
            if deg > LEV_OPEN:
                p = p + [0.0, 0.0, 0.015]
            out.append(self.pose(t, self.P(*_tcp_for_tips(p, PUSH_ROT)), SHUT, lab, PUSH_ROT))
        out.append(self.pose(1.0, self.P(*(_tcp_for_tips(p, PUSH_ROT) + [0, 0, 0.03])), SHUT, "up off the lever", PUSH_ROT))
        return out

    def collar_off(self) -> list:
        g = np.array([0.0, 0.0, 0.0])  # TCP at the collar's axis height: the fingertips close across its middle
        nom = self.P(BX, COL_Y0 + 0.006, BZ)
        out = [self.pose(2.0, self.at_part("collar", (0.0, self.s * 0.006, 0.10), nom), OPEN_F, "over the collar, open", COLLAR_ROT),
               self.pose(1.5, self.at_part("collar", (0.0, self.s * 0.006, 0.004), nom), OPEN_F, "fingers either side of the collar", COLLAR_ROT),
               self.pose(0.8, lambda: self.cmd, COLLAR_SQ, "grip the collar", COLLAR_ROT)]
        off_y = END_Y + COL_L / 2 + 0.025
        out.append(self.pose(3.0, lambda: self.cmd + self.s * np.array([0.0, 1.0, 0.0]) * (off_y - COL_Y0), COLLAR_SQ, "slide the collar off the sleeve end", COLLAR_ROT))
        out.append(self.pose(2.0, lambda: self.cmd + [0.0, 0.0, 0.05], COLLAR_SQ, "lift it clear", COLLAR_ROT))
        out.append(self.follow(2.0, "collar", g, self.P(REST_X, off_y, BZ + 0.05), COLLAR_SQ, "back, clear of the sleeve end", COLLAR_ROT))
        out.append(self.follow(2.0, "collar", g, self.P(REST_X, REST_Y, REST_Z + 0.05), COLLAR_SQ, "over the collar rest", COLLAR_ROT))
        out.append(self.follow(1.5, "collar", g, self.P(REST_X, REST_Y, REST_Z + 0.004), COLLAR_SQ, "into the rest", COLLAR_ROT))
        out.append(self.pose(0.8, lambda: self.cmd, OPEN_F, "let go", COLLAR_ROT))
        out.append(self.pose(1.5, lambda: self.cmd + [0.0, 0.0, 0.18], OPEN_F, "up, high over the 1.25 lb plate's rim", COLLAR_ROT))
        return out

    def plate_on(self) -> list:
        grip = np.array([0.0, 0.0, BIG["r"] - 0.032])  # TCP 3 cm below the plate's top rim: the pads cover its upper 4.5 cm
        nom = self.P(TRAY_X, TRAY_Y, FLOOR + BIG["r"])
        thread_y = END_Y + BIG["t"] / 2 + 0.012
        final_y = SHOULDER_Y + SMALL["t"] + BIG["t"] / 2 - 0.010
        axis = lambda y: self.P(BX, y, BZ - (HOLE_R - SLEEVE_R))  # noqa: E731
        out = [self.pose(2.5, self.at_part("big", grip + [0, 0, 0.08], nom), OPEN_F, "over the 2.5 lb plate", PLATE_ROT),
               self.pose(1.5, self.at_part("big", grip, nom), OPEN_F, "fingers either side of its top rim", PLATE_ROT),
               self.pose(0.8, lambda: self.cmd, PLATE_SQ, "grip the plate", PLATE_ROT),
               self.pose(1.5, lambda: self.cmd + [0.0, 0.0, 0.09], PLATE_SQ, "lift it out of the tray", PLATE_ROT),
               self.follow(3.0, "big", grip, self.P(OUT_X, thread_y + 0.01, BZ + SLEEVE_R + BIG["r"] - 0.005), PLATE_SQ, "out past the sleeve's end, in front of the bar", PLATE_ROT),
               self.follow(3.0, "big", grip, axis(thread_y + 0.01) + [0, 0, 0.02], PLATE_SQ, "forward, beyond the sleeve end", PLATE_ROT),
               self.follow(1.5, "big", grip, axis(thread_y), PLATE_SQ, "hole in line with the sleeve", PLATE_ROT),
               self.follow(1.0, "big", grip, axis(thread_y), PLATE_SQ, "line it up", PLATE_ROT),
               self.follow(1.0, "big", grip, axis(thread_y), PLATE_SQ, "line it up", PLATE_ROT)]
        out.append(self.pose(3.0, lambda: self.cmd + self.s * np.array([0.0, 1.0, 0.0]) * (final_y - thread_y), PLATE_SQ, "slide it on against the first plate", PLATE_ROT))
        out.append(self.pose(0.8, lambda: self.cmd, OPEN_F, "let go", PLATE_ROT))
        out.append(self.pose(1.2, lambda: self.cmd + [-0.02, 0.0, 0.07], OPEN_F, "up off the plate", PLATE_ROT))
        return out

    def collar_on(self) -> list:
        g = np.array([0.0, 0.0, 0.0])
        nom = self.P(REST_X, REST_Y, REST_Z)
        thread_y = END_Y + COL_L / 2 + 0.015
        final_y = SHOULDER_Y + SMALL["t"] + BIG["t"] + COL_L / 2 - 0.016
        axis = lambda y: self.P(BX, y, BZ)  # noqa: E731  (the bore centred on the sleeve)
        out = [self.pose(2.0, self.at_part("collar", (0.0, self.s * 0.006, 0.08), nom), OPEN_F, "over the collar in its rest", COLLAR_ROT),
               self.pose(1.5, self.at_part("collar", (0.0, self.s * 0.006, 0.004), nom), OPEN_F, "fingers either side of it", COLLAR_ROT),
               self.pose(0.8, lambda: self.cmd, COLLAR_SQ, "grip the collar", COLLAR_ROT),
               self.pose(1.2, lambda: self.cmd + [0.0, 0.0, 0.06], COLLAR_SQ, "lift it", COLLAR_ROT),
               self.follow(2.0, "collar", g, self.P(REST_X, thread_y + 0.005, BZ + 0.03), COLLAR_SQ, "out past the sleeve end, behind the bar", COLLAR_ROT),
               self.follow(2.0, "collar", g, axis(thread_y) + [0, 0, 0.02], COLLAR_SQ, "forward, beyond the sleeve end", COLLAR_ROT),
               self.follow(1.2, "collar", g, axis(thread_y), COLLAR_SQ, "in line with the sleeve", COLLAR_ROT),
               self.follow(1.0, "collar", g, axis(thread_y), COLLAR_SQ, "line it up", COLLAR_ROT),
               self.follow(1.0, "collar", g, axis(thread_y), COLLAR_SQ, "line it up", COLLAR_ROT)]
        out.append(self.pose(3.5, lambda: self.cmd + self.s * np.array([0.0, 1.0, 0.0]) * (final_y - thread_y), COLLAR_SQ, "slide it on against the new plate", COLLAR_ROT))
        out.append(self.pose(0.8, lambda: self.cmd, OPEN_F, "let go", COLLAR_ROT))
        out.append(self.pose(1.0, lambda: self.cmd + [0.0, 0.0, 0.07], OPEN_F, "up", COLLAR_ROT))
        return out, final_y

    def outside(self) -> list:
        """The easy wrong act: the new plate hung on the bare sleeve outside the locked collar."""
        p = self.plate_on()
        thread_y = END_Y + BIG["t"] / 2 + 0.012
        final_y = COL_Y0 + COL_L / 2 + BIG["t"] / 2 + 0.004
        p[9] = self.pose(3.0, lambda: self.cmd + self.s * np.array([0.0, 1.0, 0.0]) * (final_y - thread_y), PLATE_SQ, "slide it on outside the collar", PLATE_ROT)
        return p

    def push_home(self) -> list:
        """Closed fingertips on the collar's outer face, below its lever, push it in against the new plate."""
        nom = self.P(BX, COL_Y0 + BIG["t"], BZ)
        tip_z = 0.018  # above the sleeve's top (12.7 mm), on the collar's face (29 mm radius)
        def at(dy, dz):
            def target():
                st = self.state("collar")
                c = nom if st is None else st.pos
                p = np.asarray(c, float) + [0.0, self.s * dy, dz]
                return _tcp_for_tips(p, PUSH_ROT)
            return target
        return [self.pose(1.5, at(COL_L / 2 + 0.015, tip_z + 0.05), SHUT, "fingertips beyond the collar's outer face", PUSH_ROT),
                self.pose(1.2, at(COL_L / 2 + 0.010, tip_z), SHUT, "down beside it", PUSH_ROT),
                self.pose(1.5, at(COL_L / 2 - 0.006, tip_z), SHUT, "push the collar in against the plate", PUSH_ROT),
                self.pose(1.0, lambda: self.cmd + self.s * np.array([0.0, 0.012, 0.0]) + [0.0, 0.0, 0.03], SHUT, "back off", PUSH_ROT)]

    def lift(self, t: float) -> Pose:
        return self.pose(t, lambda: np.maximum(self.cmd, [0.0, -9.0, TOP + 0.28]) if self.cmd is not None else self.P(BX, 0.45, 0.28),
                         OPEN_F, "hand up, open", _tilted(30.0, 0.0))

    def away(self, t: float) -> Pose:
        """Both hands forward of the bar's middle, low: the sleeve ends stay in the top camera's view."""
        return self.pose(t, self.P(AWAY[0], AWAY[1], AWAY[2]), OPEN_F, "hand away, open", _tilted(30.0, 0.0))


def demo():
    """TASK_DEMO=success (default): each arm flips its collar's lever up, slides the collar off and sets it in its rest, threads
    the 2.5 lb plate on against the 1.25 lb plate, slides the collar back against it and presses the lever flat, then moves away.
    failure: each arm hangs its plate on the bare sleeve outside the locked collar (scores 0).
    unlocked: as success, but the levers are left up (scores 0)."""
    mode = os.environ.get("TASK_DEMO", "success")
    plans = {}
    for side, s in SIDES:
        a = Arm(side, s)
        if mode == "failure":
            plans[side] = a.outside() + [a.away(3.0)]
            continue
        on, cy = a.collar_on()
        seq = a.open_lever() + a.collar_off() + a.plate_on() + on
        if mode != "unlocked":
            seq += a.push_home() + a.close_lever(cy)
        plans[side] = seq + [a.lift(1.0), a.away(3.0)]
    return plans
