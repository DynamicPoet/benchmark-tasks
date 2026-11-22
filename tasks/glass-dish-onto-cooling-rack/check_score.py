"""Offline check of the scorer on hand-built states, including the lazy routes. No simulator. Run from this folder:

    python check_score.py

Each case builds the dish's pose history and the 24 fingertip points, steps the scorer as the harness would, and checks the score.
"""

from __future__ import annotations

import math
import sys

import numpy as np

import task as t

HOME = np.array([*t.PLATE_XY, t.PLATE_TOP + 0.0003])
ON_RACK = np.array([t.RACK_XY[0], t.RACK_XY[1], t.RACK_TOP])
FAR = np.full((24, 3), 5.0)  # hands far away


def rot_x(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def near_handles(pos: np.ndarray) -> np.ndarray:
    """Finger points of both hands pinching both handles of a dish at pos (8.5 mm above and below the handle's mid-plane)."""
    pts = []
    for side in (1.0, -1.0):
        for dz in (0.0085, -0.0085):
            for _ in range(6):
                pts.append(pos + np.array([0.0, side * (t._HY - 0.015), t.sa.HANDLE_Z + dz]))
    return np.array(pts)


def run(states) -> float:
    """states: list of (pos, rot, down_speed, speed, tips) per control step, fed through the task's own update()."""
    t._clear()
    for pos, rot, down, speed, tips in states:
        t.update(np.asarray(pos, float), rot, down, speed, tips)
    return t.score(None)


def hover_far(pos: np.ndarray) -> np.ndarray:
    """The left hand pinches; the right hand's points all hover 10 mm above the far handle, not around it."""
    pts = near_handles(pos)
    pts[12:] = pos + np.array([0.0, -(t._HY - 0.015), t.sa.HANDLE_Z + 0.004 + 0.010])
    return pts


def one_hand(pos: np.ndarray) -> np.ndarray:
    """Only the left hand pinches the left handle; the right hand is far away."""
    pts = near_handles(pos)
    pts[12:] = 5.0
    return pts


def path(a, b, n: int, tips_fn=None, rot=np.eye(3)):
    out = []
    for k in range(n):
        p = a + (b - a) * (k + 1) / n
        out.append((p, rot, 0.0, 0.05, tips_fn(p) if tips_fn else FAR))
    return out


def rest(p, n: int, tips=FAR, rot=np.eye(3)):
    return [(p, rot, 0.0, 0.0, tips) for _ in range(n)]


UP = HOME + [0, 0, 0.12]
OVER = ON_RACK + [0, 0, 0.12]
carry = rest(HOME, 5, near_handles(HOME)) + path(HOME, UP, 10, near_handles) + path(UP, OVER, 10, near_handles)

CASES = [
    ("do nothing", rest(HOME, 40), 0.0),
    ("full: set down on the rack, hands clear, still 1 s", carry + path(OVER, ON_RACK, 10, near_handles) + rest(ON_RACK, 25), 1.0),
    ("set down but still holding the handles", carry + path(OVER, ON_RACK, 10, near_handles) + rest(ON_RACK, 25, near_handles(ON_RACK)), 0.5),
    ("let go but not yet still for 1 s", carry + path(OVER, ON_RACK, 10, near_handles) + rest(ON_RACK, 10), 0.5),
    ("lazy hover: held over the rack to the end", carry + rest(OVER, 30, near_handles(OVER)), 0.5),
    ("lifted 2 cm then put back on the trivet", path(HOME, HOME + [0, 0, 0.02], 5) + path(HOME + [0, 0, 0.02], HOME, 5) + rest(HOME, 25), 0.25),
    ("lifted 4 cm then put back on the trivet", path(HOME, HOME + [0, 0, 0.04], 5) + path(HOME + [0, 0, 0.04], HOME, 5) + rest(HOME, 25), 0.25),
    ("dropped onto the rack from 10 cm", carry + [(OVER - [0, 0, 0.05], np.eye(3), 1.0, 1.0, FAR)] + rest(ON_RACK, 25), 0.0),
    ("tilted 25 deg in the carry, then set down", carry + rest(OVER, 2, near_handles(OVER), rot_x(25)) + path(OVER, ON_RACK, 10, near_handles)
     + rest(ON_RACK, 25), 0.0),
    ("set down level on the rack but tilted 8 deg", carry + path(OVER, ON_RACK + [0, 0, 0.01], 10, near_handles) + rest(ON_RACK + [0, 0, 0.01], 25, rot=rot_x(8)), 0.5),
    ("set down partly off the rack's side, on its wires", carry + path(OVER, ON_RACK + [0, -0.12, 0], 10, near_handles) + rest(ON_RACK + [0, -0.12, 0], 25), 0.5),
    ("set down 1 cm forward of the rack's middle (still on it)", carry + path(OVER, ON_RACK + [0.01, 0, 0], 10, near_handles) + rest(ON_RACK + [0.01, 0, 0], 25), 1.0),
    ("set down on the bare counter beside the hot plate", carry + rest(np.array([0.30, -0.30, t.TOP]), 25), 0.0),
    ("lowered into the washing-up bowl", carry + rest(np.array([*t.BOWL_XY, t.TOP + 0.08]), 25), 0.0),
    ("into the bowl, then onto the rack", carry + rest(np.array([*t.BOWL_XY, t.TOP + 0.08]), 2) + rest(ON_RACK, 25), 0.0),
    ("carried forward first at a 5 cm lift, then right (handle passes beside the bowl above its rim)",
     rest(HOME, 5, near_handles(HOME)) + path(HOME, HOME + [0, 0, 0.05], 8, near_handles)
     + path(HOME + [0, 0, 0.05], np.array([0.49, 0.0, HOME[2] + 0.05]), 10, near_handles)
     + path(np.array([0.49, 0.0, HOME[2] + 0.05]), np.array([0.49, -0.13, HOME[2] + 0.05]), 8, near_handles)
     + path(np.array([0.49, -0.13, HOME[2] + 0.05]), np.array([0.49, -0.13, t.RACK_TOP]), 8, near_handles)
     + rest(np.array([0.49, -0.13, t.RACK_TOP]), 25), 1.0),
    ("one arm only: lifted 4 cm by one handle, carried and set level on the rack, let go", rest(HOME, 5, one_hand(HOME))
     + path(HOME, UP, 10, one_hand) + path(UP, OVER, 10, one_hand) + path(OVER, ON_RACK, 10, one_hand) + rest(ON_RACK, 25), 0.5),
    ("one arm lifts and carries; the other hovers over the far handle at the top of the lift", rest(HOME, 5, one_hand(HOME))
     + path(HOME, UP, 10, one_hand) + rest(UP, 12, hover_far(UP)) + path(UP, OVER, 10, one_hand) + path(OVER, ON_RACK, 10, one_hand)
     + rest(ON_RACK, 25), 0.5),
    ("both hands pinch only 0.3 s at the top of a lift, else one hand", rest(HOME, 5, one_hand(HOME)) + path(HOME, UP, 10, one_hand)
     + rest(UP, 6, near_handles(UP)) + path(UP, OVER, 10, one_hand) + path(OVER, ON_RACK, 10, one_hand) + rest(ON_RACK, 25), 0.5),
    ("slid off the trivet onto the rack without a lift (no fall read)", path(HOME, np.array([0.49, -0.13, HOME[2]]), 20)
     + [(np.array([0.49, -0.13, t.RACK_TOP]), np.eye(3), 0.3, 0.3, FAR)] + rest(np.array([0.49, -0.13, t.RACK_TOP]), 25), 0.0),
    ("pushed along at plate height to the rack edge (no lift)", path(HOME, np.array([0.33, 0.0, HOME[2]]), 20) + rest(np.array([0.33, 0.0, HOME[2]]), 25), 0.0),
]


def main() -> int:
    bad = 0
    for name, states, want in CASES:
        got = run(states)
        ok = abs(got - want) < 1e-9
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {got:4.2f} (want {want:4.2f})  {name}  [{t._S['fail'] or 'no failure'}]")
    print(f"{len(CASES) - bad}/{len(CASES)} cases as stated")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
