"""On the station table: the cable carrier (energy chain) of a robot's linear axis mounted on its side, as for a low
installation height: it lies on one side plate, its fixed end clipped onto the bed bracket, stretched straight along -y;
beside it, the axis's carriage at mid-travel on its rail, with the moving-end bracket on its plate's -y end.

The robot folds the chain back on itself in the plane of the table, the one way its links bend, into its 38 mm loop, and
pushes the end link into the carriage bracket until the end link's upper pin clicks into the bracket's bore; the left hand
pins the lower run while the loop forms, then braces the carriage, which slides freely on its rail.

Parts: igus E2 mini B15.015.038.0 (polymer links, pitch 30.5 mm, outer width 26 mm, outer height 23 mm, bend radius
38 mm), 11 links; igus B15 one-piece polymer mounting brackets; an MGN12 rail and MGN12H carriage. The end link's pins
are rigid; the bracket's upper cheek flexes on a spring as the pin rides under it, and its lower cheek is slotted.

Run it:  inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \\
           -E control_hz=20 --scorer task_score --max-steps 2400 --instruction "<the task.json instruction>"
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # lets this file import the helpers beside it

import gatekit as gk  # noqa: E402
import mechkit as m  # noqa: E402
import poseplay  # noqa: E402,F401  (lets demo() turn the grip)
import steptime  # noqa: E402
from poseplay import Joints, Pose  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, READY  # noqa: E402
from yam_arena.demo import grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 0.0  # score() holds the end state for 1 s itself
SCORE_OVER = "end"
INSTRUCTION = ("Fold the cable carrier back on itself and snap its free end into the bracket on the carriage, keeping the "
               "carriage at its mark, then let go and move both hands clear.")

# -- sizes (metres): igus E2 mini B15.015.038.0 (delivery program datasheet) ---------------------------------------------

N = 11                   # links
P = 0.0305               # pitch
BA, HA = 0.026, 0.023    # outer width (across the side plates, here vertical) and outer height (here along x)
BI, HI = 0.015, 0.017    # inner width and height
PLATE = (BA - BI) / 2    # side plate thickness, 5.5 mm
BEND = math.radians(47.3)  # most a joint turns: 2 asin(P / 2R) for R 38 mm
LINK_MASS = 0.35 * P     # 0.35 kg/m
XC, ZC = HA / 2, BA / 2  # a link's axis: across its height, and above the table (lying on a side plate)
# The joint at the chain's moving end, as on every link of the series: the end link's side plates are cranked in at its
# free end (the "pin end"), half the wall thickness, and a 6 mm pin stands on each, flush with the chain's outer face; the
# moving-end bracket's cheeks lie over them: the upper pin clicks into the upper cheek's bore, the lower pin runs in the
# lower cheek's slot.
PIN_IN = 0.006           # the pins' centre: 6 mm in from the end link's free end
PIN_R = 0.003            # pin radius
CRANK = PLATE / 2        # the pin end's plates are inset this much; the pins stand this tall; the cheeks are this thick
INSET = 0.016            # length of the inset at the pin end
CHEEK_LEN = 0.015        # the bracket's cheeks reach this far back over the end link from its free end

# -- layout (world: x forward from the arms, y left; metres) -------------------------------------------------------------

X0 = 0.28                # the chain's outer-radius face, lying (toward the robot; its inner radius faces away)
FIXED = np.array([X0, 0.0, TOP])        # the chain's origin: the fixed end's pivot line on the table
TRAVEL = 0.15            # the carriage slides +-15 cm about its mark

LOOKS = {
    "chain": m.Look(m.rgb(0.07, 0.07, 0.08), roughness=0.7),
    "bar": m.Look(m.rgb(0.12, 0.12, 0.13), roughness=0.6),
    "bracket": m.Look(m.rgb(0.10, 0.10, 0.11), roughness=0.6),
    "alu": m.Look(m.rgb(0.72, 0.73, 0.75), metallic=0.7, roughness=0.35),
    "rail": m.Look(m.rgb(0.55, 0.56, 0.58), metallic=0.9, roughness=0.25),
    "steel": m.Look(m.rgb(0.62, 0.63, 0.65), metallic=0.9, roughness=0.3),
    "tape": m.Look(m.rgb(0.95, 0.80, 0.10), roughness=0.6),
}


def chain_fk(theta) -> list[tuple[np.ndarray, float]]:
    """Each link's (centre, heading) in the chain's frame for joint angles `theta` (radians, positive bends toward +x):
    forward kinematics in the plane of the table."""
    out, p, phi = [], np.array([XC, 0.0, ZC]), 0.0
    for k in range(N):
        phi += theta[k]
        d = np.array([math.sin(phi), -math.cos(phi), 0.0])
        out.append((p + 0.5 * P * d, phi))
        p = p + P * d
    return out


# The loop as the chain makes it when pushed: the two middle joints against their 47.3 degree stops, the joints either side
# sharing the rest of the 180 degrees (a loop of four 45 degree joints is 1.7 mm wider than this one)
LOOP = np.radians([42.7, 47.3, 47.3, 42.7])
FOLD = np.zeros(N)
FOLD[4:8] = LOOP                         # folded: links 1-4 straight, joints 5-8 the loop, links 8-11 coming back
END_C, END_PHI = chain_fk(FOLD)[-1]      # the end link folded: its centre, and turned 180 degrees about z
SEAT_Y = float(END_C[1])
X_OUT = float(END_C[0])                  # the outer run's axis, 71.9 mm out from the lower run's
PIN_Y = SEAT_Y + 0.5 * P - PIN_IN        # the bracket's bore (and slot end): where the seated end link's pins come to rest

# -- parts ---------------------------------------------------------------------------------------------------------------


def _octagon_pin(cx: float, cy: float, z0: float, z1: float, top_chamfer: float, up: bool) -> m.Mesh:
    """A 6 mm pin standing from z0 to z1 (pointing up if `up`), its free end chamfered."""
    pts = []
    for i in range(8):
        a = math.pi / 8 + i * math.pi / 4
        r = PIN_R / math.cos(math.pi / 8)
        rc = r - top_chamfer
        zf, zb = (z1, z0) if up else (z0, z1)
        zc = zf - top_chamfer if up else zf + top_chamfer
        pts += [(cx + r * math.cos(a), cy + r * math.sin(a), zb), (cx + r * math.cos(a), cy + r * math.sin(a), zc),
                (cx + rc * math.cos(a), cy + rc * math.sin(a), zf)]
    return _hull(np.array(pts))


def _link_parts(k: int) -> tuple[list, list]:
    """Link k lying on a side plate at joint zero, in the chain's frame: the two side plates (bottom and top), a crossbar
    on each radius. The end link's free end is its pin end: both plates cranked in and a pin standing on each."""
    y1, y0 = -(k - 1) * P, -k * P
    yc = (y0 + y1) / 2
    parts, cols = [], []
    for zc, side in ((PLATE / 2, "lo"), (BA - PLATE / 2, "hi")):
        if k == N:
            ym = y0 + INSET
            full = ((HA, y1 - ym, PLATE), (XC, (y1 + ym) / 2, zc))
            zi = zc + (CRANK / 2 if side == "lo" else -CRANK / 2)          # the inset half of the plate, toward the link's inside
            inset = ((HA, INSET, CRANK), (XC, (y0 + ym) / 2, zi))
            for j, (sz, c) in enumerate((full, inset)):
                parts.append((f"plate{side}{j}", m.box(sz, c), "chain"))
                cols.append(m.col_box(f"c{k}_plate{side}{j}", sz, c))
            if side == "hi":
                pin = _octagon_pin(XC, y0 + PIN_IN, BA - CRANK, BA, 0.0008, True)
            else:
                pin = _octagon_pin(XC, y0 + PIN_IN, 0.0010, CRANK, 0.0008, False)   # 1 mm clear of the table
            parts.append((f"pin_{side}", pin, "bar"))
            cols.append(m.col_hull(f"c{k}_pin_{side}", pin))
        else:
            sz, c = (HA, P - 0.001, PLATE), (XC, yc, zc)
            parts.append((f"plate{side}", m.box(sz, c), "chain"))
            cols.append(m.col_box(f"c{k}_plate{side}", sz, c))
    bar = ((HA - HI) / 2, 0.008, BI)
    for xc, side in (((HA - HI) / 4, "out"), (HA - (HA - HI) / 4, "in")):
        parts.append((f"bar_{side}", m.box(bar, (xc, yc, ZC)), "bar"))
        cols.append(m.col_box(f"c{k}_bar_{side}", bar, (xc, yc, ZC)))
    return parts, cols


def _chain() -> m.Part:
    links = []
    for k in range(1, N + 1):
        parts, cols = _link_parts(k)
        links.append(m.Link(f"l{k}", "hinge", "Z", (XC, -(k - 1) * P, ZC), 0.0, math.degrees(BEND), tuple(parts), tuple(cols),
                            mass=LINK_MASS, damping=0.0005, friction=0.0005, effort=5.0, armature=1e-5, gravity=True,
                            com=(XC, -(k - 0.5) * P, ZC), inertia=(1.1e-6, 1.0e-6, 1.2e-6), parent=None if k == 1 else f"l{k - 1}"))
    # the bed bracket, side-mounted: a polymer bracket screwed to the table round the first link's pivot
    base = ((HA + 0.008, 0.022, 0.003), (XC, 0.008, 0.0015))
    back = ((HA + 0.008, 0.006, BA), (XC, 0.014, ZC))
    top = ((HA + 0.008, 0.022, 0.004), (XC, 0.008, BA + 0.003))
    frame_parts = [("base", m.box(*base), "bracket"), ("back", m.box(*back), "bracket"), ("top", m.box(*top), "bracket")]
    frame_cols = [m.col_box("cb_back", *back), m.col_box("cb_top", *top)]
    return m.mechanism("chain", tuple(FIXED), links, LOOKS, frame_parts=frame_parts, frame_colliders=frame_cols, solver_iters=64,
                       friction=(0.35, 0.3))


def _hull(points: np.ndarray) -> m.Mesh:
    """The convex hull of points as a closed mesh, faces counter-clockwise from outside."""
    from scipy.spatial import ConvexHull
    h = ConvexHull(points)
    c = points.mean(axis=0)
    faces = []
    for f in h.simplices:
        a, b, d = points[f]
        if np.dot(np.cross(b - a, d - a), a - c) < 0:
            f = f[::-1]
        faces.append(f)
    return m.Mesh(points.astype(float), np.array(faces))


def _rail(outline, ya: float, z0: float, z1: float, ch: float = 0.001) -> m.Mesh:
    """A convex plate from an (x, y) outline, z0 to z1, its top edge along y = ya chamfered `ch`: a lead-in for a part
    sliding on from -y."""
    pts = []
    for x, y in outline:
        pts.append((x, y, z0))
        if abs(y - ya) < 1e-9:
            pts += [(x, y, z1 - ch), (x, y + ch, z1)]
        else:
            pts.append((x, y, z1))
    return _hull(np.array(pts))


CLICK_K = float(os.environ.get("TASK_CLICK_K", "1500"))   # N/m: the upper cheek's spring; the click then takes about 5 N along the chain (see fidelity)
PLATE_LO = BA + 0.007     # the carriage plate's underside
PLATE_LEN = 0.150        # the plate's length along the rail, from the bracket's front


def _axis() -> m.Part:
    """The rail on the table (frame), the carriage sliding along y on it (link "car") with the moving-end bracket under its
    plate's -y end, and the bracket's upper cheek (link "cheek"), which flexes up as the end link's pin clicks into its bore.
    The bracket is igus's one-piece polymer moving-end bracket "with bore" for B15-1 (26.5 mm wide, two screw holes 14 mm
    apart): a body screwed under the plate, and two cheeks reaching back over the end link's pin end, as the next link's
    side plates would. Frame origin: the chain's origin."""
    xo = X_OUT                                       # the outer run's axis
    rail_x = xo + 0.115                              # out of the way of the free end as it swings round
    rail = ((0.012, 0.36, 0.008), (rail_x, 0.0, 0.004))
    mark = ((0.015, 0.006, 0.0005), (rail_x + 0.035, 0.0, 0.00025))             # yellow tape beside the rail at the carriage's start
    frame_parts = [("rail", m.box(*rail), "rail"), ("mark", m.box(*mark), "tape")]
    frame_cols = [m.col_box("cf_rail", *rail)]
    y_in = SEAT_Y + 0.5 * P                          # the seated end link's free end
    ya = y_in - CHEEK_LEN                            # the cheeks' open end
    w = HA + 0.002                                   # across the chain's height; across its width the cheeks span 0.5-26 mm (igus: A + 26.5 mm, A = 0)
    block = ((0.027, 0.045, 0.010), (rail_x, 0.0, 0.003 + 0.005))                # MGN12H: 27 x 45 mm, top 13 mm up
    riser = ((0.027, 0.045, PLATE_LO - 0.013), (rail_x, 0.0, 0.013 + (PLATE_LO - 0.013) / 2))   # a 20 mm aluminium spacer
    plate = ((0.155, PLATE_LEN, 0.010), (xo + 0.0625, y_in + PLATE_LEN / 2, PLATE_LO + 0.005))   # its -y end flush with the bracket's body
    body = ((w, 0.012, PLATE_LO - 0.0005), (xo, y_in + 0.006, 0.0005 + (PLATE_LO - 0.0005) / 2))  # screwed under the plate
    # the lower cheek: lying 0.5 mm off the table under the end link's inset plate, slotted from its open end for the lower
    # pin
    zl0, zl1 = 0.0005, CRANK - 0.0005                # 0.5 mm under the inset: the end link's nose dips as the upper cheek presses on it
    # the lower pin runs in a plain slot, 11 mm wide: the upper bore's countersink locates the link
    slot = 0.0055
    rail_l = _rail([(xo - w / 2, ya), (xo - slot, ya), (xo - slot, y_in), (xo - w / 2, y_in)], ya, zl0, zl1)
    rail_r = _rail([(xo + slot, ya), (xo + w / 2, ya), (xo + w / 2, y_in), (xo + slot, y_in)], ya, zl0, zl1)
    lend = ((w, y_in - (PIN_Y + slot), zl1 - zl0), (xo, (y_in + PIN_Y + slot) / 2, (zl0 + zl1) / 2))
    car_parts = [("block", m.box(*block), "steel"), ("riser", m.box(*riser), "alu"), ("plate", m.box(*plate), "alu"),
                 ("body", m.box(*body), "bracket"), ("cheek_lo_l", rail_l, "bracket"), ("cheek_lo_r", rail_r, "bracket"),
                 ("cheek_lo_end", m.box(*lend), "bracket")]
    car_cols = [m.col_box("cc_block", *block), m.col_box("cc_riser", *riser), m.col_box("cc_plate", *plate), m.col_box("cc_body", *body),
                m.col_hull("cc_cheek_lo_l", rail_l), m.col_hull("cc_cheek_lo_r", rail_r), m.col_box("cc_cheek_lo_end", *lend)]
    car = m.slider("car", "Y", -TRAVEL, TRAVEL, car_parts, car_cols, mass=0.75, friction=1.5, damping=2.0, effort=200.0)   # block, spacer and plate
    # the upper cheek: over the end link's upper inset, its bore where the seated link's upper pin comes to rest, its open
    # end chamfered underneath so the pin lifts it; a vertical slide on a spring stands for the cheek's flex
    zb = BA - CRANK + 0.0002
    bore = PIN_R + 0.0002
    yf = PIN_Y - bore
    cs, ch = 0.0025, 0.0015          # the bore's countersink underneath: 2.5 mm wider a side, 1.5 mm deep
    yr = PIN_Y + bore
    front = _hull(np.array([(x, y, z) for x in (xo - w / 2, xo + w / 2)
                            for (y, z) in ((ya, BA), (yf, BA), (yf, zb + ch), (yf - cs, zb), (ya + 0.005, zb))]))   # a 5 mm lead-in ramp
    rear = _hull(np.array([(x, y, z) for x in (xo - w / 2, xo + w / 2)
                           for (y, z) in ((yr, BA), (y_in, BA), (y_in, zb), (yr + cs, zb), (yr, zb + ch))]))
    def _side(sx: int) -> m.Mesh:
        """The bore's side beyond x = sx * bore, its underside chamfered all round the bore's mouth."""
        xi, xc, xw = xo + sx * bore, xo + sx * (bore + cs), xo + sx * w / 2
        pts = [(x, y, BA) for x in (xi, xw) for y in (yf, yr)]
        pts += [(x, y, zb + ch) for x in (xi, xw) for y in (yf, yr)]
        pts += [(x, y, zb) for x in (xc, xw) for y in (yf + cs, yr - cs)]
        return _hull(np.array(pts))

    sides = [_side(-1), _side(1)]
    cheek = m.slider("cheek", "Z", 0.0, 0.006,
                     [("front", front, "bracket")] + [(f"side{j}", sd, "bracket") for j, sd in enumerate(sides)]
                     + [("rear", rear, "bracket")],
                     [m.col_hull("ck_front", front)] + [m.col_hull(f"ck_side{j}", sd) for j, sd in enumerate(sides)]
                     + [m.col_hull("ck_rear", rear)], parent="car",
                     mass=0.002, spring=CLICK_K, damping=0.5, friction=0.0, effort=40.0)
    return m.mechanism("axis", tuple(FIXED), [car, cheek], LOOKS, frame_parts=frame_parts, frame_colliders=frame_cols, solver_iters=32,
                       friction=(0.35, 0.3))


OBJECTS = [_chain(), _axis()]

# -- what the scorer measures ------------------------------------------------------------------------------------------

HOLD_STEPS = 21          # 1 s at 20 Hz
SEAT_POS = 0.003         # the end link's centre within 3 mm of its seat in the bracket
SEAT_DEG = 10.0          # and within 10 degrees of the seat's attitude
LATCH_LIFT = 0.001      # the upper pin is in its bore when the cheek is lifted less than 1 mm (1.75 mm or more of the pin in it)
CAR_KEPT = 0.010         # the carriage within 10 mm of its mark at the end
CAR_PEAK = 0.025         # and never more than 25 mm from it: a braced carriage gives a little under the push (the arm is soft), an unbraced one is shoved off
NEAR = 0.035             # the 0.25 tier: the folded end link's tip at the bracket's mouth (its centre within 35 mm behind its seat)
CLEAR = 0.03             # both hands 3 cm clear of the chain and the bracket (a line up through its bore)
TWIST = 15.0             # no link's bend axis more than 15 degrees off vertical (the chain turned about its length)

_S: dict = {}
_ENV: dict = {"env": None}


def _clear() -> None:
    _S.clear()
    _S.update(step=0, hold=0, near=False, last={}, m={}, end=gk.EndHold(HOLD_STEPS), twisted=gk.Latch("a link tipped off its side plate"),
              car_peak=gk.Peak())


_clear()


def _np(x) -> np.ndarray:
    return getattr(x, "torch", x).detach().cpu().numpy().astype(np.float64)


def _links(env) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each link's (centre, rotation) in the station frame."""
    base = env.unwrapped
    asset = base.scene["chain"]
    origin = _np(base.scene.env_origins[0])
    poses = _np(asset.data.body_link_pose_w)[0]
    out = []
    for k in range(1, N + 1):
        i = asset.body_names.index(f"l{k}")
        R = m.quat_matrix(poses[i, 3:7])
        # mechkit authors each link in the chain's frame, so a link's pose is where the chain's origin has gone
        out.append((poses[i, :3] - origin + R @ np.array([XC, -(k - 0.5) * P, ZC]), R))
    return out


def _gap_to_hands(tips, links, seat_c) -> float:
    """The fingertips' least distance from the chain's surface (along its whole length, both ends included) and from the
    bracket (a line up through its bore), in metres."""
    first, last = links[0], links[-1]
    ends = [first[0] - first[1] @ np.array([0.0, -0.5 * P, 0.0]), last[0] + last[1] @ np.array([0.0, -0.5 * P, 0.0])]
    chain = np.vstack([ends[0]] + [c for c, _ in links] + [ends[1]])
    pin = np.array([seat_c + np.array([0.0, 0.0, ZC]), seat_c + np.array([0.0, 0.5 * P, ZC])])
    return gk.min_distance(tips, [chain, pin], radius=ZC)


def measure(env) -> dict:
    links = _links(env)
    ax = m.state(env, "axis")
    car = ax.joint("car")
    seat_c = FIXED + END_C + np.array([0.0, car, 0.0])
    c, R = links[-1]
    seat_R = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])   # turned 180 degrees about z
    dR = R @ seat_R.T
    ang = math.degrees(math.acos(float(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0))))
    pin_in = abs(ax.joint("cheek")) < LATCH_LIFT
    twist = max(math.degrees(math.acos(float(np.clip(Rk[2, 2], -1.0, 1.0)))) for _, Rk in links)
    tips = m.fingertips(env)
    clear = _gap_to_hands(tips, links, seat_c)
    speed = max(float(np.linalg.norm(lc - _S["last"].get(k, lc))) for k, (lc, _) in enumerate(links))
    _S["last"] = {k: lc.copy() for k, (lc, _) in enumerate(links)}
    pin = c + R @ np.array([0.0, -(0.5 * P - PIN_IN), 0.0])     # the end link's pins' axis (its free end is its local -y)
    bore = FIXED + np.array([X_OUT, PIN_Y + car, 0.0])
    pin_dx, pin_dy = (pin[0] - bore[0]) * 1000, (pin[1] - bore[1]) * 1000
    right_tips = np.asarray(tips)[12:] if len(tips) == 24 else np.asarray(tips)
    grip_dy = float(np.mean(right_tips[:, 1]) - c[1]) * 1000          # the right hand's fingertips along the chain, from the end link's centre
    return {"grip_dy": grip_dy, "pin_dx": pin_dx, "pin_dy": pin_dy, "seat_mm": float(np.linalg.norm(c - seat_c)) * 1000, "behind_mm": float(seat_c[1] - c[1]) * 1000, "seat_deg": ang,
            "pins": int(pin_in), "car_mm": car * 1000, "twist_deg": twist, "clear_mm": clear * 1000, "move_mm": speed * 1000,
            "end": [round(float(v), 4) for v in c]}


def _seated(q: dict) -> bool:
    return q["seat_mm"] < SEAT_POS * 1000 and q["seat_deg"] < SEAT_DEG


def _kept(S: dict, q: dict) -> bool:
    return S["car_peak"].value <= CAR_PEAK * 1000 and abs(q["car_mm"]) <= CAR_KEPT * 1000


def evaluate(S: dict) -> float:
    q = S.get("m") or {}
    if not q or S["twisted"]:
        return 0.0
    released = q["clear_mm"] > CLEAR * 1000
    latched = _seated(q) and q["pins"] == 1
    if latched and released and _kept(S, q):
        return 1.0 if S["end"].done else 0.75
    if latched and released:
        return 0.75
    if _seated(q) and released:
        return 0.5
    return 0.25 if S["near"] else 0.0


def track(S: dict, q: dict) -> None:
    S["step"] += 1
    S["m"] = q
    if q["behind_mm"] < NEAR * 1000 and q["seat_mm"] < NEAR * 1000 + 5.0 and q["seat_deg"] < 15.0:
        S["near"] = True
    full = (_seated(q) and q["pins"] == 1 and q["clear_mm"] > CLEAR * 1000 and _kept(S, q)
            and q["twist_deg"] <= TWIST and q["move_mm"] < 0.5)
    S["twisted"].trip(q["twist_deg"] > TWIST, at=S["step"])
    S["car_peak"].update(abs(q["car_mm"]))
    S["end"].step(full)
    S["hold"] = S["end"].count


@m.on_reset
def _reset(env) -> None:
    """Failures stay failed until a real reset."""
    _clear()
    _ENV["env"] = env


@m.on_step
def _step(env) -> None:
    _ENV["env"] = env
    q = measure(env)
    track(_S, q)
    if os.environ.get("TASK_SCORE_DIR"):
        folder = pathlib.Path(os.environ["TASK_SCORE_DIR"])
        folder.mkdir(parents=True, exist_ok=True)
        ax = m.state(env, "axis")
        ch = m.state(env, "chain")
        row = {"step": _S["step"], "score": evaluate(_S), "hold": _S["hold"], **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in q.items()},
               "joints": [round(math.degrees(ch.joint(f"l{k}")), 1) for k in range(1, N + 1)],
               "pin_mm": round(1000 * ax.joint("cheek"), 2)}
        with open(folder / "track.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")


def score(objects) -> float:
    return evaluate(_S)


def success(objects) -> bool:
    steptime.tick()  # the harness calls success() once per control step
    return evaluate(_S) == 1.0


# -- the scripted demo -------------------------------------------------------------------------------------------------

MODE = os.environ.get("TASK_DEMO", "success")
FIRM = -0.0090           # 2.5 mm inside the link's 23 mm height: about 6 N a finger
OVER = float(os.environ.get("TASK_OVER", "0.045"))       # how far past the seat the push is commanded: the arm gives under the load
PRELOAD = float(os.environ.get("TASK_PRELOAD", "0.004"))  # how far the bracing fingers are commanded into the carriage plate's end
PUSH = -0.0035           # jaws 16 mm under the link's 23 mm height: about 40 N a finger, the YAM's figure, to push the click through the pinch
DOWN_L = grip_rotation(-math.pi / 2)   # the left hand straight down, fingers closing along x (across a lying link's height)
TCP_UP = 0.002           # the TCP 2 mm above a lying link's axis (15 mm up): the fingers' undersides about 7 mm off the table


def _right_rot(heading: float) -> np.ndarray:
    """The right hand straight down, fingers closing across the link's height, for a link turned `heading` about z."""
    return grip_rotation(-math.pi / 2 + heading)


def roll(f: float) -> np.ndarray:
    """Joint angles with the loop rolled from joints 6-9 (f = 0) to joints 5-8 (f = 1, the seated fold): the end moves on
    about two pitches, its heading staying 180 degrees."""
    th = np.zeros(N)
    th[5:9] += (1.0 - f) * LOOP
    th[4:8] += f * LOOP
    return th


def link_at(th, k: int) -> tuple[np.ndarray, float]:
    """Link k's centre (station frame) and heading for joint angles th."""
    c, phi = chain_fk(th)[k - 1]
    return FIXED + c, phi


def fold_path(n: int = 24, m_roll: int = 10) -> list[tuple[np.ndarray, float]]:
    """The end link's centre (station frame) and heading along the fold, all on the table. First joints 6 to 9 bend
    together from straight into the loop: the end swings round toward +x and comes back along the outer run, two
    pitches short of its seat, links 1 to 5 straight (the left hand pins link 4). Then the bend rolls back half a joint
    (joint 9 straightening as joint 5 bends), carrying the end one pitch on, to just short of the bracket's mouth; the
    outer run comes out a few millimetres on the way (the polygon effect)."""
    out = []
    for f in np.linspace(0.0, 1.0, n + 1)[1:]:
        th = np.zeros(N)
        th[5:9] = f * LOOP
        out.append(link_at(th, N))
    out += [link_at(roll(f), N) for f in np.linspace(0.0, 0.5, m_roll + 1)[1:]]
    return out


def _on_link(k: int, nominal: np.ndarray, dz: float, dy: float = 0.0):
    def f(plan):
        c = _link_now(k)
        return tuple((nominal if c is None else c) + np.array([0.0, dy, dz]))
    return f


def _link_now(k: int):
    env = _ENV["env"]
    if env is None:
        return None
    return _links(env)[k - 1][0]


def _shift(d):
    d = np.asarray(d, float)

    def f(plan):
        return tuple(np.asarray(plan._hp["pos"], float) + d)
    return f


def _push(d, x: float):
    """Like _shift, but with the TCP's x set to `x`."""
    d = np.asarray(d, float)

    def f(plan):
        p = np.asarray(plan._hp["pos"], float) + d
        p[0] = x
        return tuple(p)
    return f


def _right() -> list:
    path = fold_path()
    start = link_at(np.zeros(N), N)[0]
    pre = path[-1][0]                                  # the end link one pitch short of its seat
    r0, r1 = _right_rot(0.0), _right_rot(math.pi)
    l11 = link_at(roll(0.5), N)[0]
    BACK = -0.5 * P - 0.004                            # the push grip straddles the last hinge, centred 4 mm onto link 10: it holds both
    # links straight (pushed through a free hinge the end link folds aside), and the fingers stay 4 mm off the bracket's cheek
    w = [
        Pose(1.5, tuple(start + np.array([-0.02, 0.0, 0.10])), FINGER_OPEN, "toward the free end", rot=r0),
        Pose(1.0, _on_link(N, start, 0.06), FINGER_OPEN, "above the free end", rot=r0),
        Pose(1.2, _on_link(N, start, TCP_UP), FINGER_OPEN, "fingers either side of the end link", rot=r0),
        Pose(0.8, _on_link(N, start, TCP_UP), FIRM, "grip the end link", rot=r0),
    ]
    up = np.array([0.0, 0.0, TCP_UP])
    w += [Pose(0.5, tuple(q + up), FIRM, "swing the end round, turning the wrist", rot=_right_rot(phi)) for q, phi in path[:24]]
    w += [Pose(0.4, tuple(q + up), FIRM, "roll the loop on", rot=r1) for q, _ in path[24:]]
    w += [
        Pose(0.6, tuple(pre + up), FINGER_OPEN, "let go of the end link", rot=r1),
        Pose(0.8, tuple(pre + np.array([0.0, -0.01, 0.05])), FINGER_OPEN, "up", rot=r1),
        Pose(1.0, _on_link(N, l11, 0.05, BACK), FINGER_OPEN, "over the last hinge", rot=r1),
        Pose(1.2, _on_link(N, l11, TCP_UP, BACK), FINGER_OPEN, "fingers either side of it", rot=r1),
        Pose(0.8, _shift((0.0, 0.0, 0.0)), PUSH, "grip it hard", rot=r1),
        Pose(1.5, _shift((0.0, 0.0, 0.0)), PUSH, "wait for the carriage to be braced", rot=r1),
    ]
    # the push: link 10 along its rolled path (in a few mm and on one pitch), then on into the stop: the arm lags under load
    # x follows the rolled path absolutely, ending on the bracket's line; y and z go on from where the grip was taken
    fs = np.linspace(0.5, 1.0, 7)
    for f0, f1 in zip(fs[:-1], fs[1:]):
        d = link_at(roll(f1), N)[0] - link_at(roll(f0), N)[0]
        w.append(Pose(0.5, _push(d, link_at(roll(f1), N)[0][0]), PUSH, "push the end link into the bracket", rot=r1))
    x_seat = float((FIXED + END_C)[0])
    w += [
        Pose(2.0, _push((0.0, OVER, 0.0), x_seat), PUSH, "push on until it clicks: the arm gives under the load", rot=r1),
        Pose(0.8, _push((0.0, -OVER, 0.0), x_seat), PUSH, "ease off", rot=r1),
        Pose(0.8, _shift((0.0, 0.0, 0.0)), FINGER_OPEN, "let go", rot=r1),
        Pose(1.0, _shift((0.0, 0.0, 0.06)), FINGER_OPEN, "straight up", rot=r1),
        Pose(0.8, _shift((0.0, -0.03, 0.0)), FINGER_OPEN, "rise clear", rot=r1),
        Joints(2.0, (0.0, 0.0, 0.0), FINGER_OPEN, "back to ready", q=tuple(READY)),
    ]
    return w


def _left(t_pin_off: float, t_brace: float, t_free: float, total: float) -> list:
    """The left hand pins link 4 to the table, so the chain bends at its far end and not at its fixed end; once the end has
    swung half round, it lifts off and stands its closed fingers against the carriage plate's +y end, so the push cannot shove
    the carriage along its rail; it stays until the right hand has let go."""
    pin = FIXED + chain_fk(np.zeros(N))[3][0] + np.array([0.0, 0.004, TCP_UP])   # 4 mm toward link 3: clear of link 5 as it turns
    plate_y = SEAT_Y + 0.5 * P + PLATE_LEN + 0.009 - PRELOAD   # the carriage plate's +y face, half the closed fingers, less the preload
    brace = FIXED + np.array([X_OUT + 0.035, plate_y, PLATE_LO + 0.005 + 0.008])   # fingers' undersides at the plate's mid-height
    hi = 0.02
    return [
        Pose(1.5, tuple(FIXED + np.array([-0.06, -0.02, 0.15])), FINGER_OPEN, "high, in front of the chain", rot=DOWN_L),
        Pose(1.5, tuple(pin + np.array([0.0, 0.0, 0.06])), FINGER_OPEN, "over link 4", rot=DOWN_L),
        Pose(1.6, tuple(pin), FINGER_OPEN, "fingers either side of link 4", rot=DOWN_L),
        Pose(0.8, tuple(pin), FIRM, "pin link 4 to the table", rot=DOWN_L),
        Pose(max(0.5, t_pin_off - 5.4), tuple(pin), FIRM, "hold the lower run", rot=DOWN_L),
        Pose(0.5, tuple(pin), FINGER_OPEN, "let go of link 4", rot=DOWN_L),
        Pose(0.7, tuple(pin + np.array([0.0, 0.012, hi])), FINGER_OPEN, "up off it", rot=DOWN_L),
        Pose(2.0, (pin[0] + 0.02, FIXED[1] + 0.06, pin[2] + 0.06), FINGER_CLOSED, "back along the chain, away from the other hand", rot=DOWN_L),
        Pose(max(1.0, t_brace - t_pin_off - 3.2 - 1.5), tuple(brace + np.array([0.0, 0.0, 0.06])), FINGER_CLOSED, "over the carriage's end", rot=DOWN_L),
        Pose(1.5, tuple(brace), FINGER_CLOSED, "fingers against the carriage plate's end", rot=DOWN_L),
        Pose(max(0.5, t_free - t_brace), tuple(brace), FINGER_CLOSED, "brace the carriage", rot=DOWN_L),
        Pose(1.2, tuple(brace + np.array([0.0, 0.02, 0.05])), FINGER_CLOSED, "away", rot=DOWN_L),
        Joints(2.0, (0.0, 0.0, 0.0), FINGER_OPEN, "back to ready", q=tuple(READY)),
    ]


def demo():
    """The left hand pins link 4; the right hand grips the end link and swings it round on the table, turning the wrist as
    the link turns, until it lies on the outer run with its tip in the bracket's mouth; it lets go, grips the next link,
    and once the left hand braces the carriage, pushes the end link home over the pin, rolling the bend back a pitch.
    TASK_DEMO=none: both arms wait. TASK_DEMO=failure: the push without the brace. Check modes: lazy_early_out lets go
    by the bracket without pushing; lazy_hands keeps hold of the end link after the click."""
    wait = lambda t: [Joints(t, (0.0, 0.0, 0.0), FINGER_OPEN, "wait", q=tuple(READY))]  # noqa: E731
    if MODE == "none":
        return {"left": wait(10.0), "right": wait(10.0)}
    right = _right()
    total = sum(x.duration for x in right)
    k_off = 4 + 12                                   # the end has swung half round: the left hand lets go of link 4 before the right comes back near it
    k_push = next(i for i, x in enumerate(right) if x.label.startswith("push"))
    k_go = next(i for i, x in enumerate(right) if x.label == "let go")
    t_pin_off = sum(x.duration for x in right[:k_off])
    t_brace = sum(x.duration for x in right[:k_push]) - 0.2
    t_free = sum(x.duration for x in right[:k_go + 1]) + 0.3
    left = _left(t_pin_off, t_brace, t_free, total)
    if MODE == "failure":
        left = left[:8] + [Joints(max(0.5, total - sum(x.duration for x in left[:8])), (0.0, 0.0, 0.0), FINGER_OPEN, "back to ready", q=tuple(READY))]
    elif MODE == "lazy_early_out":                   # lets go once the end lies by the bracket, never pushes it in: 0.25 at most
        k_up = next(i for i, x in enumerate(right) if x.label == "up")
        right = right[:k_up + 1] + [Joints(2.0, (0.0, 0.0, 0.0), FINGER_OPEN, "back to ready", q=tuple(READY))]
    elif MODE == "lazy_hands":                       # pushes it in and keeps hold of it: not released, 0.25 at most
        right = right[:k_go] + [Pose(4.0, _shift((0.0, 0.0, 0.0)), PUSH, "keep hold", rot=_right_rot(math.pi))]
    return {"left": left, "right": right}
