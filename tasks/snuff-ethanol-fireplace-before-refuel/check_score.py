"""Offline checks of the scorer on hand-built states (no simulator): python check_score.py, with the harness's Python, from this folder.

Each case feeds the task's own Trial a sequence of per-step states (fireplace, tube, tool, bottle, 24 fingertip points, arm link points),
as the simulation does every 50 ms, and reads its score at the end. Exits non-zero if a milestone or a lazy route scores wrongly.
"""

from __future__ import annotations

import math
import sys

import numpy as np

import gatekit as gk
import task as T

TOP = T.TOP
LEVEL_HAND = np.column_stack([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])  # jaws close along y, fingers point +x
FAR_LINKS = np.tile(np.array([0.0, -0.30, TOP + 0.10]), (10, 1))


def fire():
    return gk.Pose((T.FX, T.FY, TOP))


def tube(dz: float = 0.0, dx: float = 0.0):
    return gk.Pose((T.FX + dx, T.FY, TOP + T.STONES_Z + dz))


def bottle(cap_deg: float = 0.0, tilt_deg: float = 0.0):
    R = gk.rot((1.0, 0.0, 0.0), tilt_deg) if tilt_deg else None
    return gk.Pose((T.BX, T.BY, TOP), R=R, joints={"cap_joint": math.radians(cap_deg)})


def tool(x, y, z):
    return gk.Pose((x, y, TOP + z))


STAND = (T.TX, T.TY, 0.0)
OUT = (T.FX + 0.002, T.FY, T.CUP_Z + 0.0005)


def pinch(tp, opening: float = T.STRIP_W / 2):
    """The right hand pinching the strip of a tool at `tp` across its width (left hand parked)."""
    mid = np.array([tp.pos[0], tp.pos[1], tp.pos[2] + T.GRIP_UP])
    return gk.tips(right=gk.hand(mid, LEVEL_HAND, opening))


def pinch_sim(tp):
    """The pinch as the simulator reports it: the hand's tool point on the strip, so the fingertip rows lie 13 mm ahead of it and 17 mm
    behind it along the fingers (mechkit's points are at 147 and 117 mm, the tool point at 134 mm)."""
    mid = np.array([tp.pos[0] + 0.013, tp.pos[1], tp.pos[2] + T.GRIP_UP])
    return gk.tips(right=gk.hand(mid, LEVEL_HAND, T.STRIP_W / 2))


def hand_at(p, opening: float = 0.03):
    return gk.tips(right=gk.hand(np.asarray(p, float), LEVEL_HAND, opening))


PARKED = gk.tips()


def run(steps) -> float:
    """steps: a list of (fireplace, tube, tool, bottle, tips) per 50 ms step."""
    tr = T.Trial()
    for f, g, t, b, tips in steps:
        tr.observe(f, g, t, b, tips, FAR_LINKS)
    return tr.score


def seq(*parts):
    out = []
    for n, make in parts:
        out += [make(k) for k in range(n)]
    return out


def lift_path(k, n, a, b):
    s = min(1.0, k / max(1, n - 1))
    return tool(*(np.asarray(a) + s * (np.asarray(b) - np.asarray(a))))


def held_step(t):
    return (fire(), tube(), t, bottle(), pinch(t))


NOTHING = seq((60, lambda k: (fire(), tube(), tool(*STAND), bottle(), PARKED)))
PICK = seq((10, lambda k: (fire(), tube(), tool(*STAND), bottle(), PARKED)),
           (30, lambda k: held_step(lift_path(k, 30, STAND, (T.TX, T.TY, 0.10)))))
OUT_HELD = PICK + seq((40, lambda k: held_step(lift_path(k, 40, (T.TX, T.TY, 0.10), OUT))))
OUT_LET_GO = OUT_HELD + seq((40, lambda k: (fire(), tube(), tool(*OUT), bottle(), hand_at((OUT[0] - 0.035, OUT[1], OUT[2] + T.GRIP_UP + TOP)))))
DONE = OUT_HELD + seq((50, lambda k: (fire(), tube(), tool(*OUT), bottle(), PARKED)))

MILESTONES = [("the tool picked up by its strip", PICK), ("its disc on the cup, still held", OUT_HELD),
              ("let go, a hand within 5 cm", OUT_LET_GO), ("let go, hands clear for 1 s", DONE)]


def drop_case():
    """Carried over the tube and let go 21 cm up: the disc falls onto the cup and lands covering it."""
    over = (T.FX, T.FY, T.CUP_Z + 0.21)
    s = PICK + seq((20, lambda k: held_step(lift_path(k, 20, (T.TX, T.TY, 0.10), over))))
    fall = [over[2] - 0.5 * 9.81 * (k * 0.05) ** 2 for k in range(1, 6)]
    s += [(fire(), tube(), tool(over[0], over[1], max(z, OUT[2])), bottle(), PARKED) for z in fall]
    s += seq((50, lambda k: (fire(), tube(), tool(*OUT), bottle(), PARKED)))
    return s


ROUTES = [
    ("an open hand round the strip, never closed", seq((10, lambda k: (fire(), tube(), tool(*STAND), bottle(), PARKED)),
                                                        (40, lambda k: (fire(), tube(), tool(*STAND), bottle(), pinch(tool(*STAND), 0.03)))), 0.0),
    ("the tool dropped down the tube onto the cup", drop_case(), 0.0),
    ("a finger inside the tube while setting the disc down", OUT_HELD + seq((5, lambda k: (fire(), tube(), tool(*OUT), bottle(),
                                                                                         hand_at((T.FX, T.FY, TOP + 0.20))))) + DONE[-50:], 0.0),
    ("the fuel cap turned 40 degrees, then the flame put out", seq((10, lambda k: (fire(), tube(), tool(*STAND), bottle(40.0), PARKED))) + DONE, 0.0),
    ("the fuel bottle tipped 50 degrees", seq((10, lambda k: (fire(), tube(), tool(*STAND), bottle(0.0, 50.0), PARKED))) + DONE, 0.0),
    ("the glass tube lifted 2 cm and set back", seq((10, lambda k: (fire(), tube(), tool(*STAND), bottle(), PARKED)),
                                                     (10, lambda k: (fire(), tube(0.02), tool(*STAND), bottle(), PARKED))) + DONE, 0.0),
    ("the fireplace pushed 3 cm, then the flame put out", seq((10, lambda k: (fire(), tube(), tool(*STAND), bottle(), PARKED)),
                                                              (10, lambda k: (gk.Pose((T.FX + 0.03, T.FY, TOP)), tube(dx=0.03), tool(*STAND), bottle(), PARKED)))
     + DONE, 0.0),
    ("the disc set 8 mm off centre", PICK + seq((50, lambda k: (fire(), tube(), tool(T.FX + 0.008, T.FY, OUT[2]), bottle(), PARKED))), 0.25),
    ("the disc held 1 cm over the cup", PICK + seq((50, lambda k: held_step(tool(T.FX, T.FY, T.CUP_Z + 0.01)))), 0.25),
]


def sim_cases() -> list[str]:
    """Cases built with the simulator's finger geometry (the tool point on the strip)."""
    lift = seq((10, lambda k: (fire(), tube(), tool(*STAND), bottle(), PARKED)),
               (30, lambda k: (lambda t: (fire(), tube(), t, bottle(), pinch_sim(t)))(lift_path(k, 30, STAND, (T.FX, T.FY, 0.30)))))
    fast = lift + seq((6, lambda k: (lambda t: (fire(), tube(), t, bottle(), pinch_sim(t)))(lift_path(k, 6, (T.FX, T.FY, 0.30), OUT))),
                      (40, lambda k: (fire(), tube(), tool(*OUT), bottle(), pinch_sim(tool(*OUT)))))
    out = []
    for name, case, want in (("picked up and held in the air", lift, 0.25),
                             ("lowered at about 0.7 m/s while pinched, kept pinched on the cup", fast, 0.5)):
        got = run(case)
        print(f"{'ok ' if got == want else 'BAD'} {got:.4f}  simulator geometry: {name} (want {want})")
        if got != want:
            out.append(f"simulator-geometry case '{name}' scores {got}, not {want}")
    return out


def main() -> int:
    problems = gk.check_ladder(run, NOTHING, MILESTONES, ROUTES) + sim_cases()
    for p in problems:
        print("PROBLEM:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
