"""Offline check of the scorer on hand-built states, with the lazy routes. Run from this folder with the harness's Python:

    python check_score.py

It needs no simulator: each case feeds `Trial.observe` the parts' poses and the fingertip points for a number of steps.
"""

from __future__ import annotations

import sys

import numpy as np

import task as t

AX = t.TOP + t.BZ  # the sleeve axis height (station frame)


class S:
    """A part's state as the scorer reads it."""

    def __init__(self, pos, lever: float = 0.0, R=None):
        self.pos = np.asarray(pos, float)
        self._R = np.eye(3) if R is None else R
        self._lever = lever

    def rotation(self):
        return self._R

    def tilt_deg(self) -> float:
        return float(np.degrees(np.arccos(np.clip(self._R[2, 2], -1, 1))))

    def deg(self, name: str) -> float:
        return self._lever


def hang(hole_r: float) -> float:
    return AX - (hole_r - t.SLEEVE_R)


def start_states() -> dict:
    out = {}
    for side, s in t.SIDES:
        out[f"small_{side}"] = S((t.BX, s * t.SMALL_Y, hang(t.HOLE_R)))
        out[f"collar_{side}"] = S((t.BX, s * t.COL_Y0, hang(t.BORE_R)), 0.0)
        out[f"big_{side}"] = S((t.TRAY_X, s * t.TRAY_Y, t.TOP + t.FLOOR + t.BIG["r"]))
    return out


def done_end(states: dict, side: str, s: float, lever: float = 0.0, plate_gap: float = 0.002, collar_gap: float = 0.002) -> None:
    by = t.SHOULDER_Y + t.SMALL["t"] + plate_gap + t.BIG["t"] / 2
    cy = by + t.BIG["t"] / 2 + collar_gap + t.COL_L / 2
    states[f"big_{side}"] = S((t.BX, s * by, hang(t.HOLE_R)))
    states[f"collar_{side}"] = S((t.BX, s * cy, hang(t.BORE_R)), lever)


FAR = np.tile([[0.18, 0.0, t.TOP + 0.45]], (24, 1))  # every fingertip well away
NEAR = np.tile([[t.BX, t.SHOULDER_Y + 0.05, AX + 0.03]], (24, 1))  # fingertips on the left sleeve


def run(states_seq, tips=FAR) -> float:
    tr = t.Trial()
    for st in states_seq:
        tr.observe(st, tips)
    return tr.score


def steady(states: dict, n: int = 60) -> list:
    return [states] * n


def case_full():
    st = start_states()
    for side, s in t.SIDES:
        done_end(st, side, s)
    return run(steady(st))


def case_one_end():
    st = start_states()
    done_end(st, "left", 1.0)
    return run(steady(st))


def case_nothing():
    return run(steady(start_states()))


def case_outside():
    st = start_states()
    for side, s in t.SIDES:
        y = t.COL_Y0 + t.COL_L / 2 + t.BIG["t"] / 2 + 0.003
        st[f"big_{side}"] = S((t.BX, s * y, hang(t.HOLE_R)))
    return run(steady(st))


def case_lever_up():
    st = start_states()
    for side, s in t.SIDES:
        done_end(st, side, s, lever=40.0)
    return run(steady(st))


def case_collars_off():
    st = start_states()
    for side, s in t.SIDES:
        done_end(st, side, s)
        st[f"collar_{side}"] = S((t.REST_X, s * t.REST_Y, t.TOP + t.REST_Z), 45.0)
    return run(steady(st))


def case_hands_near():
    st = start_states()
    for side, s in t.SIDES:
        done_end(st, side, s)
    return run(steady(st), tips=NEAR)


def case_collar_gap():
    st = start_states()
    for side, s in t.SIDES:
        done_end(st, side, s, collar_gap=0.008)
    return run(steady(st))


def case_plate_gap():
    st = start_states()
    for side, s in t.SIDES:
        done_end(st, side, s, plate_gap=0.015)
    return run(steady(st))


def case_dragged():
    """A clamped collar slid 2 cm out along the sleeve (forced), then the right state is reached anyway."""
    a = start_states()
    b = start_states()
    b["collar_left"] = S((t.BX, t.COL_Y0 + 0.02, hang(t.BORE_R)), 0.0)
    c = start_states()
    for side, s in t.SIDES:
        done_end(c, side, s)
    return run(steady(a, 8) + steady(b, 3) + steady(c, 60))


def case_dropped():
    """A plate let fall to the table, then the end state is reached with it put back."""
    a = start_states()
    a["big_left"] = S((0.20, 0.55, t.TOP + 0.01))
    c = start_states()
    for side, s in t.SIDES:
        done_end(c, side, s)
    return run(steady(start_states(), 8) + steady(a, 3) + steady(c, 60))


def case_not_still():
    """Both ends done but the left plate still swinging 3 mm a step."""
    seq = []
    for k in range(60):
        st = start_states()
        for side, s in t.SIDES:
            done_end(st, side, s)
        p = st["big_left"].pos.copy()
        p[0] += 0.003 * (k % 2)
        st["big_left"] = S(p)
        seq.append(st)
    return run(seq)


def case_plate_on_edge():
    """A plate let fall that lands upright on its edge on the table, then is put on properly."""
    a = start_states()
    a["big_left"] = S((0.30, 0.55, t.TOP + t.BIG["r"]))
    c = start_states()
    for side, s in t.SIDES:
        done_end(c, side, s)
    return run(steady(start_states(), 8) + steady(a, 3) + steady(c, 60))


def case_collar_dropped():
    """A collar dropped onto the table beside the bar, then fitted."""
    a = start_states()
    a["collar_left"] = S((0.35, 0.55, t.TOP + t.COL_R), 45.0)
    c = start_states()
    for side, s in t.SIDES:
        done_end(c, side, s)
    return run(steady(start_states(), 8) + steady(a, 3) + steady(c, 60))


def case_order_swapped():
    """The 2.5 lb plate pushed in against the shoulder, inside the 1.25 lb plate."""
    st = start_states()
    for side, s in t.SIDES:
        done_end(st, side, s)
        by = t.SHOULDER_Y + t.BIG["t"] / 2
        sy = by + t.BIG["t"] / 2 + 0.002 + t.SMALL["t"] / 2
        st[f"big_{side}"] = S((t.BX, s * by, hang(t.HOLE_R)))
        st[f"small_{side}"] = S((t.BX, s * sy, hang(t.HOLE_R)))
    return run(steady(st))


def case_half_open_drag():
    """The lever nudged up 20 degrees (still gripping), and the collar slid off."""
    a = start_states()
    b = start_states()
    b["collar_left"] = S((t.BX, t.COL_Y0 + 0.02, hang(t.BORE_R)), 20.0)
    c = start_states()
    for side, s in t.SIDES:
        done_end(c, side, s)
    return run(steady(a, 8) + steady(b, 3) + steady(c, 60))


def case_plates_swapped():
    """Both ends done, but each 2.5 lb plate taken from the other side's tray (they are identical)."""
    st = start_states()
    for side, s in t.SIDES:
        done_end(st, side, s)
    st["big_left"], st["big_right"] = st["big_right"], st["big_left"]
    return run(steady(st))


def case_swapped_outside():
    """The other side's 2.5 lb plate hung outside this side's locked collar."""
    st = start_states()
    y = t.COL_Y0 + t.COL_L / 2 + t.BIG["t"] / 2 + 0.003
    st["big_right"] = S((t.BX, y, hang(t.HOLE_R)))
    return run(steady(st))


CASES = [
    ("both ends done, hands away, still", case_full, 1.0),
    ("one end done, the other untouched", case_one_end, 0.5),
    ("nothing done", case_nothing, 0.0),
    ("plates hung outside the locked collars", case_outside, 0.0),
    ("collars refitted, levers left up", case_lever_up, 0.0),
    ("plates inside, collars left in their rests", case_collars_off, 0.0),
    ("both ends done, a hand still on the sleeve", case_hands_near, 0.5),
    ("collars locked 8 mm from the plates", case_collar_gap, 0.0),
    ("new plates 15 mm from the first", case_plate_gap, 0.0),
    ("a clamped collar dragged along the sleeve", case_dragged, 0.0),
    ("a plate dropped to the table", case_dropped, 0.0),
    ("both ends done, a plate still moving", case_not_still, 0.5),
    ("a plate dropped onto its edge on the table", case_plate_on_edge, 0.0),
    ("a collar dropped on the table", case_collar_dropped, 0.0),
    ("the 2.5 lb plate inside the 1.25 lb plate", case_order_swapped, 0.0),
    ("a collar slid off with its lever only 20 degrees up", case_half_open_drag, 0.0),
    ("both ends done with the plates from the other trays", case_plates_swapped, 1.0),
    ("the other side's plate hung outside a locked collar", case_swapped_outside, 0.0),
]


def main() -> int:
    bad = 0
    for name, fn, want in CASES:
        got = fn()
        ok = abs(got - want) < 1e-9
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {got:.2f} (want {want:.2f})  {name}")
    print(f"{len(CASES) - bad}/{len(CASES)} cases pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
