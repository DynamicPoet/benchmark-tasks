"""Offline checks of the scorer's ladder and geometry on hand-built records, with the harness's Python and no simulator: python check_score.py"""

import math
import sys

import numpy as np

import importlib.util  # noqa: E402


def rec(**kw):
    s = dict(T=50.0, zero=False, why="", bowl_steps=0, run=0, cooled=False, served_ok=False, done=False, n=0)
    s.update(kw)
    return s


def main() -> None:
    spec = importlib.util.spec_from_file_location("task", "task.py")
    t = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t)
    cases = {
        "nothing": (rec(), 0.0),
        "10 s in the bowl, hot": (rec(bowl_steps=200), 0.25),
        "9.9 s in the bowl": (rec(bowl_steps=199, cooled=False), 0.0),
        "cooled, left in the bowl": (rec(bowl_steps=4000, cooled=True), 0.5),
        "served cool, still held": (rec(bowl_steps=4000, cooled=True, served_ok=True), 0.75),
        "served cool, let go": (rec(bowl_steps=4000, cooled=True, served_ok=True, done=True), 1.0),
        "served hot (zero)": (rec(bowl_steps=4000, cooled=True, served_ok=True, done=True, zero=True), 0.0),
    }
    bad = 0
    for name, (s, want) in cases.items():
        got = t.evaluate(s)
        print(f"{'ok ' if got == want else 'BAD'} {name}: {got} (want {want})")
        bad += got != want
    # geometry: the bottle standing upright on the bowl's floor is in the water, wet to 57 mm; tipped about 41 deg or more its collar meets the water
    I = np.eye(3)
    w = np.array([0.0, 0.0, 0.0])
    inside, wet, collar = t.in_bowl(np.array([0.0, 0.0, t.W_FLOOR]), I, w, I)
    print("upright on the floor:", inside, round(wet * 1000, 1), "mm wet, collar wet", collar)
    bad += not (inside and abs(wet - 0.057) < 1e-3 and not collar)
    a = math.radians(60)
    R = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    _, _, collar = t.in_bowl(np.array([0.0, 0.0, t.W_FLOOR]), R, w, I)
    print("tipped 60 deg in the bowl: collar wet", collar)
    bad += not collar
    # lazy routes
    sq = np.array([t.SQUARE[0], t.SQUARE[1], t.TOP + t.TRAY_TOP])
    print("upright on the square:", t.on_square(sq, I), "| tipped 20 deg:", t.on_square(sq, R20 := np.array(
        [[math.cos(0.35), 0, math.sin(0.35)], [0, 1, 0], [-math.sin(0.35), 0, math.cos(0.35)]])), "| held 3 cm above:", t.on_square(sq + [0, 0, 0.03], I))
    bad += not (t.on_square(sq, I) and not t.on_square(sq, R20) and not t.on_square(sq + [0, 0, 0.03], I))
    hover = sq + np.array([0.0, 0.0, 0.03])
    print("hot bottle hovered 3 cm over the tray is on the tray:", t.on_tray(hover))
    _, wet_hover, _ = t.in_bowl(np.array([0.0, 0.0, t.W_FLOOR + 0.05]), I, w, I)
    print("held 5 cm off the bowl's floor:", round(wet_hover * 1000, 1), "mm wet; on the floor:", t.on_floor(np.array([0.0, 0.0, t.W_FLOOR + 0.05]), I, w, I))
    bad += t.on_floor(np.array([0.0, 0.0, t.W_FLOOR + 0.05]), I, w, I)
    teat_near_face = t.HEAD - np.array([t.HEAD_R + 0.05, 0.0, 0.0])
    print("teat 5 cm from the face is near the baby:", t.near_baby(teat_near_face, 0.10))
    bad += not t.near_baby(teat_near_face, 0.10)
    print("tau_w on the floor:", round(t.tau_water(0.057)), "s; to 37 C:", round(t.tau_water(0.057) * math.log(45 / 32)), "s")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
