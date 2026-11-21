"""Offline checks of the scorer on hand-built states (no simulator): python check_score.py, with the harness's Python, from this folder.

Each case feeds the task's own Trial a sequence of per-step states (cooker, lid, 24 fingertip points, arm points), as the simulation
does every 50 ms, and reads its score at the end. Exits non-zero if a milestone or a lazy route scores wrongly.
"""

from __future__ import annotations

import math
import sys

import numpy as np

import gatekit as gk
import task as T

TOP = T.TOP
FAR = np.tile(np.array([0.0, -0.30, TOP + 0.10]), (10, 1))
YAW = gk.rot((0.0, 0.0, 1.0), T.YAW)
VENT_STEPS = int(T.P_START / T.P_RATE / 0.05) + 2


def cooker(dx: float = 0.0, tilt: float = 0.0):
    R = YAW @ gk.rot((1.0, 0.0, 0.0), tilt) if tilt else YAW
    return gk.Pose((T.CX + dx, T.CY, TOP), R=R)


def lid(turn: float = 0.0, handle: float = 0.0, at=None, lift: float = 0.0):
    """The lid: on the cooker turned `turn` degrees, or at `at` (x, y, height of its origin above the table)."""
    pos = (T.CX, T.CY, TOP + T.LID_Z + lift) if at is None else (at[0], at[1], TOP + at[2])
    return gk.Pose(pos, R=gk.rot((0.0, 0.0, 1.0), turn) @ YAW, joints={"release_joint": math.radians(handle)})


def on_handle(ls, opening: float = T.HANDLE[1] / 2):
    """The left hand gripping the lid handle from above."""
    mid = ls.to_world(np.array([[0.0, 0.0, T.GRIP_Z]]))[0]
    return gk.tips(left=gk.hand(mid, None, opening))


def over_valve(ls):
    mid = ls.to_world(np.array([[T.VALVE[0], T.VALVE[1], T.CAP_TOP + 0.02]]))[0]
    return gk.tips(right=gk.hand(mid, None, 0.0))


def on_pin(ls):
    mid = ls.to_world(np.array([[T.PIN[0], T.PIN[1], T.DOME_TOP + T.PIN_UP]]))[0]
    return gk.tips(right=gk.hand(mid, None, 0.0))


PARKED = gk.tips()
IN_SQUARE = (T.SQ_X, T.SQ_Y, T.SKIRT_DROP)


def run(steps) -> float:
    tr = T.Trial()
    for c, l, tips in steps:
        tr.observe(c, l, tips, FAR)
    return tr.score


def seq(*parts):
    out = []
    for n, make in parts:
        out += [make() for _ in range(n)]
    return out


START = (10, lambda: (cooker(), lid(), PARKED))
VENT = (VENT_STEPS, lambda: (cooker(), lid(handle=-45.0), PARKED))
TURN = (10, lambda: (cooker(), lid(turn=31.0, handle=-45.0), on_handle(lid(turn=31.0))))
LIFT = (10, lambda: (cooker(), lid(turn=31.0, handle=-45.0, lift=0.06), on_handle(lid(turn=31.0, lift=0.06))))
PLACED_HELD = (30, lambda: (cooker(), lid(turn=31.0, handle=-45.0, at=IN_SQUARE), on_handle(lid(turn=31.0, at=IN_SQUARE))))
PLACED = (30, lambda: (cooker(), lid(turn=31.0, handle=-45.0, at=IN_SQUARE), PARKED))

CASES = [
    ("nothing", seq(START, (100, lambda: (cooker(), lid(), PARKED))), 0.0),
    ("the locked lid turned to the end of its play", seq(START, (5, lambda: (cooker(), lid(turn=1.4), on_handle(lid(turn=1.4))))), 0.0),
    ("venting, a hand resting on the lid handle, the lid not turned", seq(START, (20, lambda: (cooker(), lid(turn=0.3, handle=-45.0), on_handle(lid(turn=0.3))))), 0.25),
    ("handle to Venting, steam leaving", seq(START, (20, lambda: (cooker(), lid(handle=-45.0), PARKED))), 0.25),
    ("handle only half way (20 degrees)", seq(START, (200, lambda: (cooker(), lid(handle=-20.0), PARKED))), 0.0),
    ("vented and waited, lid left shut", seq(START, VENT, (20, lambda: (cooker(), lid(handle=-45.0), PARKED))), 0.5),
    ("vented, lid turned while steam still leaves", seq(START, (100, lambda: (cooker(), lid(handle=-45.0), PARKED)), TURN), 0.0),
    ("vented, a hand over the valve while steam leaves", seq(START, (20, lambda: (cooker(), lid(handle=-45.0), over_valve(lid())))), 0.0),
    ("a hand over the valve after the steam has stopped", seq(START, VENT, (20, lambda: (cooker(), lid(handle=-45.0), over_valve(lid())))), 0.5),
    ("the float valve touched while up", seq(START, (5, lambda: (cooker(), lid(), on_pin(lid())))), 0.0),
    ("vented, waited, lid turned and lifted off", seq(START, VENT, TURN, LIFT), 0.75),
    ("lid in the square, still held", seq(START, VENT, TURN, LIFT, PLACED_HELD), 0.75),
    ("lid in the square, hands clear, held 1.5 s", seq(START, VENT, TURN, LIFT, PLACED_HELD, PLACED), 1.0),
    ("lid set down outside the square", seq(START, VENT, TURN, LIFT, (30, lambda: (cooker(), lid(turn=31.0, handle=-45.0, at=(0.25, 0.30, T.SKIRT_DROP)), PARKED))), 0.75),
    ("the cooker dragged 3 cm", seq(START, VENT, TURN, LIFT, (30, lambda: (cooker(dx=0.03), lid(turn=31.0, handle=-45.0, at=IN_SQUARE), PARKED))), 0.75),
    ("the cooker tipped over", seq(START, VENT, (5, lambda: (cooker(tilt=20.0), lid(handle=-45.0), PARKED))), 0.0),
    ("the lid let fall from the lift", seq(START, VENT, TURN, LIFT, (1, lambda: (cooker(), lid(turn=31.0, handle=-45.0, lift=0.0), PARKED))), 0.0),
]


def main() -> int:
    bad = 0
    for name, steps, want in CASES:
        got = run(steps)
        ok = abs(got - want) < 1e-9
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {got:.2f} (want {want:.2f})  {name}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
