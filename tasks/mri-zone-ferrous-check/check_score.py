"""Offline checks of the scorer on hand-built states (no simulator). Run with the harness's Python from this folder:

  python check_score.py HARNESS_DIR
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import task as t  # noqa: E402

I3 = np.eye(3)
FAR_TIPS = np.full((24, 3), [0.0, 0.0, 1.5])          # every fingertip high above the table, clear of everything
NO_LINKS = np.zeros((0, 3))
HELD_Z = t.TEST_Z                                     # the magnet lifted, as when it tests a basin


def basin(xy, z=t.BASIN_Z, yaw=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    return (np.array([xy[0], xy[1], z]), np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]), 0.0)


def magnet(face_xy=t.MAG_FACE, z=t.MAG_Z):
    return (np.array([face_xy[0], face_xy[1], z]), I3, 0.0)


A0, B0, C0, M0 = basin(t.XY["a"]), basin(t.XY["b"]), basin(t.XY["c"], yaw=t.YAW["c"]), magnet()
A_ON = basin(t.A_TARGET, z=t.TROLLEY_FLOOR + 0.0005)
M_UP = magnet(z=HELD_Z)


def st(a=A0, b=B0, c=C0, m=M0, gap_a=0.05, gap_b=0.05, gap_c=0.05, tips=FAR_TIPS, links=NO_LINKS, n=1):
    return [(tips, links, {"a": a, "b": b, "c": c}, m, {"a": gap_a, "b": gap_b, "c": gap_c})] * n


def run(steps):
    t._clear()
    for s in steps:
        t.track(t._S, *s)
    return t.evaluate(t._S)


TEST_A = st(m=M_UP, gap_a=0.001, n=12)
TEST_B = st(m=M_UP, gap_b=0.0, n=12)
TEST_C = st(m=M_UP, gap_c=0.001, n=12)
END = st(a=A_ON, n=45)


def main() -> None:
    results, expect = {}, {}

    def case(name, steps, want):
        results[name], expect[name] = run(steps), want

    case("nothing", st(n=40), 0.0)
    case("full: A and B tested, A placed", TEST_B + st(m=M_UP) + TEST_A + END, 1.0)
    c_on = basin(t.A_TARGET, z=t.TROLLEY_FLOOR + 0.0005)
    case("C tested, not grabbed, sent in", TEST_C + TEST_B + st(c=c_on, n=45), 0.0)
    case("C sent in beside a placed A", TEST_B + TEST_A + st(a=A_ON, c=basin((t.A_TARGET[0], t.A_TARGET[1] + 0.12)), n=45), 0.0)
    case("only C tested, nothing sent in", TEST_C + st(n=30), 0.25)
    half_c = basin((t.TROLLEY_OUT[0] - 0.02, -0.10))                         # C's middle off, its +x side over the trolley
    case("C half over the trolley at the end", TEST_B + TEST_A + st(a=A_ON, c=half_c, n=45), 0.75)
    c_tips = FAR_TIPS.copy()
    c_tips[:] = [t.XY["c"][0], t.XY["c"][1], t.BASIN_Z + 0.02]               # a hand left in basin C
    case("a hand left in basin C", TEST_B + TEST_A + st(a=A_ON, tips=c_tips, n=45), 0.75)
    case("A placed, B never tested", TEST_A + END, 0.75)
    case("full but held under 1 s", TEST_B + TEST_A + st(a=A_ON, n=25), 0.75)
    case("A placed untested", TEST_B + END, 0.0)
    case("touch under 0.5 s counts not", st(m=M_UP, gap_a=0.001, n=6) + TEST_B + END, 0.0)
    case("bump of the lying magnet counts not", st(gap_a=0.001, gap_b=0.001, n=20), 0.0)
    case("tested, kept out", TEST_B + TEST_A + st(n=30), 0.25)
    case("B sent in too", TEST_B + TEST_A + st(a=A_ON, b=basin((t.A_TARGET[0], t.A_TARGET[1] + 0.14)), n=30), 0.0)
    over = basin(t.A_TARGET, z=t.TOP + 0.10)
    case("B carried over and back", TEST_B + TEST_A + st(b=over, n=3) + END, 0.0)
    case("magnet sent in", TEST_B + TEST_A + st(a=A_ON, m=magnet((t.A_TARGET[0], t.A_TARGET[1])), n=30), 0.0)
    tips = FAR_TIPS.copy()
    tips[0] = [t.LINE_FAR + 0.02, 0.0, t.TOP + 0.1]
    case("a fingertip past the doorway", TEST_B + TEST_A + st(a=A_ON, tips=tips, n=30), 0.0)
    case("an arm link past the doorway", TEST_B + TEST_A + st(a=A_ON, links=np.array([[t.LINE_FAR + 0.05, 0.0, 1.0]]), n=30), 0.0)
    near_b = magnet((t.XY["b"][0], t.XY["b"][1] + t.sa.BASIN_B.length / 2 + 0.002), z=HELD_Z)
    case("B left on the magnet", TEST_B + TEST_A + st(a=A_ON, m=near_b, n=45), 0.75)
    tipped = (np.array([t.A_TARGET[0], t.A_TARGET[1], t.TROLLEY_FLOOR + 0.04]),
              np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float), 0.0)
    case("A tipped on the trolley", TEST_B + TEST_A + st(a=tipped, n=30), 0.5)
    in_bowl = FAR_TIPS.copy()
    in_bowl[:] = [t.A_TARGET[0], t.A_TARGET[1], t.TROLLEY_FLOOR + 0.006]     # a closed hand resting in basin A
    case("a hand resting in basin A", TEST_B + TEST_A + st(a=A_ON, tips=in_bowl, n=45), 0.5)
    half_b = basin((t.TROLLEY_OUT[0] - 0.02, 0.0))                           # B's middle off, its +x side over the trolley
    case("B half over the trolley at the end", TEST_B + TEST_A + st(a=A_ON, b=half_b, n=45), 0.75)
    moving = [st(a=(np.array([t.A_TARGET[0], t.A_TARGET[1] + 0.001 * k, t.TROLLEY_FLOOR + 0.0005]), I3, 0.0))[0]
              for k in range(45)]
    case("A still creeping on the trolley", TEST_B + TEST_A + moving, 0.5)
    bad = 0
    for k, v in results.items():
        ok = abs(v - expect[k]) < 1e-9
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {k:38s} {v:.2f} (expected {expect[k]:.2f})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
