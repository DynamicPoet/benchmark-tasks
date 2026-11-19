"""Offline check of the scorer on hand-built states: no simulator. Run: python check_score.py"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import task as t  # noqa: E402

BASE = {"seat_mm": 0.8, "behind_mm": 0.5, "seat_deg": 2.0, "pins": 1, "car_mm": 3.0, "twist_deg": 1.0, "clear_mm": 80.0, "move_mm": 0.1}


def run(q: dict, steps: int = 30, near: bool = False) -> float:
    t._clear()
    if near:
        t.track(t._S, dict(BASE, seat_mm=30.0, behind_mm=30.0, seat_deg=5.0, pins=0, clear_mm=10.0))
    for _ in range(steps):
        t.track(t._S, q)
    return t.evaluate(t._S)


CASES = [
    ("latched, carriage kept, released, still 1 s", BASE, 30, False, 1.0),
    ("the same, held only 10 steps", BASE, 10, False, 0.75),
    ("latched but carriage moved 25 mm", dict(BASE, car_mm=25.0), 30, False, 0.75),
    ("seated, pin not in its bore", dict(BASE, pins=0), 30, False, 0.5),
    ("latched, hand still on it", dict(BASE, clear_mm=10.0), 30, True, 0.25),
    ("twisted chain, latched", dict(BASE, twist_deg=40.0), 30, False, 0.0),
    ("tip reached the mouth, then pulled away", dict(BASE, seat_mm=150.0, behind_mm=150.0, seat_deg=170.0, pins=1), 30, True, 0.25),
    ("nothing done", dict(BASE, seat_mm=300.0, behind_mm=300.0, seat_deg=180.0, pins=1, car_mm=0.0), 30, False, 0.0),
]


def tipped_then_flat() -> float:
    t._clear()
    t.track(t._S, dict(BASE, twist_deg=40.0))
    for _ in range(30):
        t.track(t._S, BASE)
    return t.evaluate(t._S)


def shoved_30_then_back() -> float:
    t._clear()
    t.track(t._S, dict(BASE, car_mm=30.0, clear_mm=5.0))
    for _ in range(30):
        t.track(t._S, BASE)
    return t.evaluate(t._S)


def shoved_then_back() -> float:
    t._clear()
    t.track(t._S, dict(BASE, car_mm=18.0, clear_mm=5.0))
    for _ in range(30):
        t.track(t._S, BASE)
    return t.evaluate(t._S)


CASES_FN = [("a link tipped onto its edge mid-run, flat and latched at the end", tipped_then_flat, 0.0),
            ("carriage shoved 18 mm during the push, back at its mark at the end", shoved_then_back, 1.0),
            ("carriage shoved 30 mm during the push, back at its mark at the end", shoved_30_then_back, 0.75)]
bad = 0
for name, fn, want in CASES_FN:
    got = fn()
    ok = abs(got - want) < 1e-9
    bad += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {got:.2f} (want {want:.2f})  {name}")
for name, q, steps, near, want in CASES:
    got = run(q, steps, near)
    ok = abs(got - want) < 1e-9
    bad += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {got:.2f} (want {want:.2f})  {name}")
sys.exit(1 if bad else 0)
