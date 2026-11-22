"""Offline score check: the task's scorer (Trial.observe) on hand-built states, including the lazy routes. No simulator.

    python check_score.py
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import task as t  # noqa: E402

CH_POS = np.array([t.CHARGER_XY[0], t.CHARGER_XY[1], t.TOP + t.MAT_T])
CH_R = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # yaw -90: the charger's +y along world +x
FAR = np.array([[0.20, 0.25, t.TOP + 0.20]] * 12 + [[0.20, -0.25, t.TOP + 0.20]] * 12)


def axis_rot(direction) -> np.ndarray:
    """A rotation whose z axis (the cell's + end) points along `direction`."""
    z = np.asarray(direction, float) / np.linalg.norm(direction)
    x = np.cross([0.0, 0.0, 1.0], z) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def on_table(name):
    kind, x, y, toward = t.CELLS[name]
    return np.array([x, y, t.TOP + t.CELL_R]), axis_rot([1.0 if toward else -1.0, 0.0, 0.0])


def in_bay(bay, right_way=True, lift=0.0):
    local = np.array([t.BAYS_X[bay], t.SEAT_Y, t.SEAT_Z + lift])
    d = CH_R @ np.array([0.0, 1.0 if right_way else -1.0, 0.0])
    return CH_POS + CH_R @ local, axis_rot(d)


def run(states, tips=FAR, steps=90):
    """`states`: a function of the step giving name -> (pos, R)."""
    tr = t.Trial()
    tr.reset()
    for k in range(steps):
        s = states(k)
        tr.observe(s, CH_POS, CH_R, tips(k) if callable(tips) else tips)
    return tr.score


def layout(**override):
    def at(k):
        s = {n: on_table(n) for n in t.CELLS}
        for n, v in override.items():
            s[n] = v(k) if callable(v) else v
        return s
    return at


CASES = [
    ("nothing done", layout(), FAR, 0.0),
    ("one eneloop seated", layout(nimh_a=in_bay(2)), FAR, 0.5),
    ("both eneloops seated, lithium out, hands away", layout(nimh_a=in_bay(2), nimh_b=in_bay(0)), FAR, 1.0),
    ("both seated, a hand still on a cell", layout(nimh_a=in_bay(2), nimh_b=in_bay(0)),
     np.vstack([np.repeat([in_bay(2)[0] + [0, 0, 0.012]], 12, 0), FAR[12:]]), 0.0),
    ("one eneloop seated backwards, the other right", layout(nimh_a=in_bay(2, False), nimh_b=in_bay(0)), FAR, 0.5),
    ("both eneloops backwards", layout(nimh_a=in_bay(2, False), nimh_b=in_bay(0, False)), FAR, 0.0),
    ("literal: all four loaded", layout(nimh_a=in_bay(0), li_b=in_bay(1), nimh_b=in_bay(2), li_a=in_bay(3)), FAR, 0.0),
    ("lithium seated 0.25 s then taken out, both eneloops seated",
     layout(nimh_a=in_bay(2), nimh_b=in_bay(0), li_b=lambda k: in_bay(3) if k < 6 else on_table("li_b")), FAR, 0.0),
    ("lithium seated for one step only, then out, both eneloops seated",
     layout(nimh_a=in_bay(2), nimh_b=in_bay(0), li_b=lambda k: in_bay(3) if k < 1 else on_table("li_b")), FAR, 0.0),
    ("one eneloop seated, a lithium cell lying unseated in a bay (propped 4 mm up)",
     layout(nimh_a=in_bay(2), li_b=in_bay(0, lift=0.004)), FAR, 0.0),
    ("lithium carried 5 cm over the charger for 1 s, both eneloops seated",
     layout(nimh_a=in_bay(2), nimh_b=in_bay(0), li_b=lambda k: in_bay(3, lift=0.05) if k < 20 else on_table("li_b")), FAR, 1.0),
    ("lithium left lying on top of the charger", layout(nimh_a=in_bay(2), nimh_b=in_bay(0),
                                                      li_b=(CH_POS + np.array([0, 0, t.HUMP_H + t.CELL_R]), axis_rot([1, 0, 0])))
     , FAR, 0.0),
    ("eneloop resting on the lip, not down", layout(nimh_a=in_bay(2, lift=0.004), nimh_b=in_bay(0)), FAR, 0.5),
    ("both eneloops in one bay's place (impossible stack)", layout(nimh_a=in_bay(2), nimh_b=in_bay(2)), FAR, 0.0),
    ("a cell off the table", layout(nimh_a=in_bay(2), nimh_b=in_bay(0), li_a=(np.array([0.22, 0.16, t.TOP - 0.1]), axis_rot([1, 0, 0]))),
     FAR, 0.0),
]


def main() -> int:
    bad = 0
    for name, states, tips, want in CASES:
        got = run(states, tips)
        ok = math.isclose(got, want)
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {got:.2f} (want {want:.2f})  {name}")
    print("all pass" if not bad else f"{bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
