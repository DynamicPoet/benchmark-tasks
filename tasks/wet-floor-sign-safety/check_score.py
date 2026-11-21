"""Offline check of the scorer on hand-built states, no simulator: doing nothing, each tier, and the lazy routes.

Run with the harness's Python, from this folder: python check_score.py
"""

from __future__ import annotations

import math
import sys

import numpy as np
from scipy.spatial.transform import Rotation

import mechkit as m
import task as t

FAR_TIPS = np.array([[0.30, 0.55 if i < 12 else -0.55, 1.0] for i in range(24)])  # both hands well away
FAR_LINKS = np.array([[0.0, 0.305, 0.80], [0.0, -0.305, 0.80], [0.10, 0.40, 0.95], [0.10, -0.40, 0.95]])


def part(pos, rot: Rotation, joint: float | None = None) -> m.PartState:
    p = np.asarray(pos, float)
    joints = {} if joint is None else {"panel_b_joint": float(joint)}
    return m.PartState(pos=p, quat_xyzw=rot.as_quat(), lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=p, high=p, joints=joints)


def posted(x=0.443, y=0.297, yaw_deg=90.0, joint=t.OPEN, lean_deg=0.0) -> m.PartState:
    """The sign standing on its feet, opened to `joint`, its panels' normals along y for yaw 90; `lean_deg` tips it over its feet."""
    half = joint / 2
    hinge_z = t.H(t.P * math.cos(half + math.radians(lean_deg)))
    rot = Rotation.from_euler("z", yaw_deg, degrees=True) * Rotation.from_euler("y", -math.degrees(half) - lean_deg, degrees=True)
    return part((x, y, hinge_z), rot, joint)


HANGING = part(t.SIGN_HINGE, Rotation.identity(), 0.0)
CART_START = part((t.CART_START[0], t.CART_START[1], t.FLOOR), Rotation.identity())
CART_PARKED = part((0.41, -0.749, t.FLOOR), Rotation.identity())
CART_HALF = part((0.41, -0.50, t.FLOOR), Rotation.identity())


def hanging_on(cart: m.PartState) -> m.PartState:
    d = np.array(t.SIGN_HINGE) - np.array([t.CART_START[0], t.CART_START[1], t.FLOOR])
    return part(cart.pos + d, Rotation.identity(), 0.0)


def run(sign, cart, latched, tips=FAR_TIPS, links=FAR_LINKS, steps=25, drift=None) -> float:
    """Score after `steps` control steps of the same state (the sign moved by `drift` per step, if given)."""
    tr = t.Tracker()
    tr.latched = latched
    for k in range(steps):
        s = sign if drift is None else part(sign.pos + k * np.asarray(drift), Rotation.from_quat(sign.quat_xyzw), sign.joints["panel_b_joint"])
        tr.step(s, cart, tips, links)
    return tr.score()


def tips_near(point, gap) -> np.ndarray:
    tips = FAR_TIPS.copy()
    tips[0] = np.asarray(point, float) + np.array([0.0, 0.0, gap])
    return tips


def main() -> int:
    good = posted()
    lever_top = good.to_world(np.array([[t.T, 0.0, t.LEVER]]))[0]
    face_pt = good.to_world(np.array([[t.T + 0.0, 0.0, -0.30]]))[0]
    lying = part((0.60, 0.30, t.H(0.016)), Rotation.from_euler("y", 90, degrees=True), 0.0)
    cases = [
        ("nothing: sign on its hook, cart where it was", HANGING, CART_START, False, {}, 0.0),
        ("literal: cart parked, sign still on its hook", hanging_on(CART_PARKED), CART_PARKED, False, {}, 0.0),
        ("sign laid folded on the wet floor, cart parked", lying, CART_PARKED, False, {}, 0.0),
        ("sign stood folded in the zone, cart parked", posted(joint=0.0), CART_PARKED, False, {}, 0.25),
        ("folded sign leaning 8 degrees on something", posted(x=0.60, y=0.40, joint=0.0, lean_deg=8.0), CART_PARKED, False, {}, 0.0),
        ("folded sign leaning 25 degrees on something", posted(x=0.60, y=0.40, joint=0.0, lean_deg=25.0), CART_PARKED, False, {}, 0.0),
        ("sign half open, tab not set", posted(joint=math.radians(28)), CART_PARKED, False, {}, 0.25),
        ("sign open but edge-on to the corridor", posted(x=0.60, y=0.40, yaw_deg=0.0), CART_PARKED, True, {}, 0.25),
        ("sign open, turned 45 degrees", posted(x=0.60, y=0.40, yaw_deg=45.0), CART_PARKED, True, {}, 0.25),
        ("sign posted, cart left where it was", good, CART_START, True, {}, 0.5),
        ("sign posted, cart half way to its bay", good, CART_HALF, True, {}, 0.5),
        ("sign posted outside the corridor (feet over the near side)", posted(x=0.36), CART_PARKED, True, {}, 0.0),
        ("posted and parked, a hand 3 cm from the sign", good, CART_PARKED, True, {"tips": tips_near(lever_top, 0.03)}, 0.75),
        ("posted and parked, a hand holding the handle", good, CART_PARKED, True, {"tips": tips_near(lever_top, 0.0)}, 0.0),
        ("posted and parked, a hand on a spray bottle's trigger", good, CART_PARKED, True, {"tips": tips_near(CART_PARKED.pos + np.array([0.07, -0.14, t.TRAYS[1] + 0.255]), 0.0)}, 0.75),
        ("posted and parked, a hand on the sign hook", good, CART_PARKED, True, {"tips": tips_near(CART_PARKED.pos + np.array([0.0, t.HOOK_Y, t.HOOK_H + 0.006]), 0.0)}, 0.75),
        ("posted and parked, a forearm leaning on a panel", good, CART_PARKED, True, {"links": np.vstack([FAR_LINKS, face_pt + np.array([0.0, 0.02, 0.0])])}, 0.0),
        ("posted and parked, sign still sliding 1 cm/s", good, CART_PARKED, True, {"drift": (0.0, 0.0005, 0.0)}, 0.75),
        ("posted and parked, hands clear, held 1 s", good, CART_PARKED, True, {}, 1.0),
    ]
    bad = 0
    print(f"{'case':66s} expect  got")
    for name, s, c, latched, kw, want in cases:
        got = run(s, c, latched, **kw)
        ok = abs(got - want) < 1e-9
        bad += not ok
        print(f"{name:66s} {want:5.2f} {got:5.2f} {'ok' if ok else 'FAIL'}")
    short = t.Tracker()
    short.latched = True
    for _ in range(t.HOLD_STEPS):
        short.step(good, CART_PARKED, FAR_TIPS, FAR_LINKS)
    held_short = short.score()
    print(f"{'full end state for only 0.95 s':66s}  0.75 {held_short:5.2f} {'ok' if held_short == 0.75 else 'FAIL'}")
    bad += held_short != 0.75
    print("all ok" if not bad else f"{bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
