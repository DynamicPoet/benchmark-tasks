"""Offline score check: the scorer on hand-built states, no simulator.

    python check_score.py

Each case feeds `Trial.observe` a sequence of control steps (handle angle, bin pose, fingertips) and checks the final score.
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import task as t  # noqa: E402

I3 = np.eye(3)
HOME = np.array([t.BIN_POS[0], t.BIN_POS[1], 0.0])
AWAY = np.tile(np.array([0.20, -0.30, 0.30]), (24, 1))  # every fingertip far from the bar


def on_bar(deg: float) -> np.ndarray:
    """All 24 fingertips resting on the bar's top, 3 cm out from the hinge."""
    return np.tile(t.bar_points(deg)[t.bar_points(deg)[:, 0].argmin()], (24, 1))


def tilt_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def edge_tip(deg: float) -> np.ndarray:
    """The bin's origin when it is tipped `deg` about the edge of its base on its -y side."""
    pivot = np.array([0.0, -t.BIN_R, 0.0])
    return HOME + pivot - tilt_x(deg) @ pivot


def run(steps) -> float:
    tr = t.Trial()
    for deg, pos, R, tips in steps:
        tr.observe(deg, pos, R, tips)
    return tr.score


def pull(to: float, n: int = 20):
    return [(to * k / n, HOME, I3, on_bar(to * k / n)) for k in range(1, n + 1)]


def rest(deg: float, n: int, pos=HOME, R=I3, tips=AWAY):
    return [(deg, pos, R, tips)] * n


CASES = [
    ("does nothing", rest(0.0, 40), 0.0),
    ("pulls to 19 deg, lets go", pull(-19.0) + rest(0.0, 30), 0.0),
    ("pulls half way (25 deg), springs back", pull(-25.0) + rest(0.0, 30), 0.25),
    ("pulls to the stop, latches, hands clear 1 s", pull(-40.0) + rest(-40.0, 25), 1.0),
    ("latches, hands clear only 0.5 s", pull(-40.0) + rest(-40.0, 10), 0.5),
    ("latches, hand left on the bar", pull(-40.0) + rest(-40.0, 30, tips=on_bar(-40.0)), 0.5),
    ("lifts the bin 1.2 cm, then latches", rest(0.0, 5, pos=HOME + [0, 0, 0.012]) + pull(-40.0) + rest(-40.0, 25), 0.0),
    ("tips the bin onto one edge (rim 1.5 cm up)", rest(0.0, 5, pos=edge_tip(3.1), R=tilt_x(3.1)) + pull(-40.0) + rest(-40.0, 25), 0.0),
    ("slides the bin 3.5 cm and back, then latches", rest(0.0, 5, pos=HOME + [0, 0.035, 0]) + rest(0.0, 5) + pull(-40.0) + rest(-40.0, 25), 0.0),
    ("nudges the bin 2 cm, latches", rest(0.0, 5, pos=HOME + [0.02, 0, 0]) + pull(-40.0) + rest(-40.0, 25, pos=HOME + [0.02, 0, 0]), 1.0),
    ("latches, then lifts the bin", pull(-40.0) + rest(-40.0, 25) + rest(-40.0, 5, pos=HOME + [0, 0, 0.05]), 0.0),
]


def main() -> int:
    bad = 0
    for name, steps, want in CASES:
        got = run(steps)
        ok = abs(got - want) < 1e-9
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {got} (want {want})")
    print(f"{len(CASES) - bad}/{len(CASES)} cases pass")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
