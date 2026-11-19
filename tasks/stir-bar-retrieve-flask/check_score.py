"""Offline checks of the scorer and the magnet's geometry, on hand-built states (no simulator).

Run it from this folder with the harness's Python:  python check_score.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import scene_assets as sa
import task as t


def _state(**kw) -> dict:
    s = {"upset": False, "tip_in": False, "lifted": False, "out": False, "hold": 0}
    s.update(kw)
    return s


def main() -> None:
    flask = np.array([0.30, -0.08, t.TOP])
    up = np.eye(3)
    side = np.array([1.0, 0.0, 0.0])
    # the bar drawn against the tube's +x side, the tube's tip at the neck's height
    for name, axis_off, scraped in (("centred", -t.BUNDLE, False), ("on the tube's axis", 0.0, True)):
        tip = flask + np.array([axis_off, 0.0, sa.FLASK_H - 0.01])
        bar = tip + side * (sa.TUBE_R + sa.BAR_R + t.SIDE_GAP) + np.array([0.0, 0.0, sa.BAR_L / 2])
        got = t.touches_neck(bar, up @ t._R_SIDE, (flask, up))
        assert got == scraped, (name, got)
        print(f"ok  tube and bar {name} in the neck: scraped off = {got}")
    # in the flask's belly the bar is never scraped
    tip = flask + np.array([0.0, 0.0, 0.03])
    bar = tip + side * (sa.TUBE_R + sa.BAR_R + t.SIDE_GAP) + np.array([0.0, 0.0, sa.BAR_L / 2])
    assert not t.touches_neck(bar, t._R_SIDE, (flask, np.eye(3)))
    print("ok  in the flask's belly: not scraped")
    # the side pose: lengthwise along the tube, lower end at the tip, touching it
    p = t.side_pose(np.zeros(3), np.eye(3), np.array([0.02, 0.0, 0.004]))
    assert abs(p[0] - (sa.TUBE_R + sa.BAR_R + t.SIDE_GAP)) < 1e-9 and abs(p[2] - sa.BAR_L / 2) < 1e-9
    print("ok  side pose", np.round(p, 4))
    # score ladder
    cases = [
        ("doing nothing", _state(), 0.0),
        ("tip in the flask", _state(tip_in=True), 0.25),
        ("bar lifted 3 cm", _state(tip_in=True, lifted=True), 0.5),
        ("bar above the rim", _state(tip_in=True, lifted=True, out=True), 0.75),
        ("end state held 1 s", _state(tip_in=True, lifted=True, out=True, hold=t.HOLD_STEPS), 1.0),
        ("end state 0.5 s", _state(tip_in=True, lifted=True, out=True, hold=10), 0.75),
        ("flask tipped, bar poured out", _state(upset=True, tip_in=False, lifted=True, out=True, hold=t.HOLD_STEPS), 0.0),
    ]
    for name, s, want in cases:
        got = t.evaluate(s)
        assert got == want, (name, got, want)
        print(f"ok  {name}: {got}")


if __name__ == "__main__":
    main()


def lazy_routes() -> None:
    """Routes that skip part of the act, on hand-built states; each must miss the 1.0 state."""
    import math

    good = dict(in_beaker=True, attached=False, release_slide=True, tube_gap=0.05, tool_in=False, flask_tilt=0.0,
                beaker_tilt=0.0, hand_gap=0.2, bar_speed=0.0)
    assert t.end_state(**good)
    print("ok  the real end state counts")
    cases = [
        ("bar scraped off on the beaker's rim, magnet never slid up", dict(release_slide=False)),
        ("bar still on the magnet over the beaker", dict(attached=True)),
        ("retriever left standing in the beaker", dict(tool_in=True)),
        ("retriever lying against the bar", dict(tube_gap=0.004)),
        ("a fingertip next to the bar", dict(hand_gap=0.01)),
        ("bar dropped beside the beaker", dict(in_beaker=False)),
        ("beaker knocked over with the bar in it", dict(beaker_tilt=math.radians(40))),
        ("flask left leaning 15 degrees", dict(flask_tilt=math.radians(15))),
        ("bar still rolling", dict(bar_speed=0.05)),
    ]
    for name, change in cases:
        s = dict(good)
        s.update(change)
        assert not t.end_state(**s), name
        print(f"ok  no 1.0: {name}")
    # the magnet does not draw the bar through the glass
    flask = np.array([0.30, -0.08, t.TOP])
    bar = flask + np.array([0.0, 0.0, sa.FLOOR + sa.BAR_R])
    F = (flask, np.eye(3))
    assert t.inside_flask(bar, F)
    outside = flask + np.array([sa.FLASK_R + sa.TUBE_R + 0.001, 0.0, 0.02])
    assert not t.inside_flask(outside, F)
    print("ok  a tube outside the flask's wall cannot pick the bar up through the glass")
    in_neck = flask + np.array([0.0, 0.0, sa.FLASK_H - 0.02])
    assert t.inside_flask(in_neck, F)
    print("ok  a tip down the neck is inside the flask")
    under = flask + np.array([0.0, 0.0, -0.005])
    assert not t.inside_flask(under, F)
    print("ok  a tip under a lifted flask's base is not inside it (no 0.25)")
    assert abs(t.BAR_CORNER - sa.BAR_R / math.cos(math.pi / 8)) < 1e-12 and t.BAR_CORNER > sa.BAR_R
    src = Path(t.__file__).read_text()
    assert "hand_gap = min(seg_dist(a, b, p, p) for p in tips) - BAR_CORNER" in src
    print("ok  the fingertip and tube clearances are measured from the octagon's corners (4.33 mm)")


if __name__ == "__main__":
    lazy_routes()


def more_routes() -> None:
    """A release away from the beaker, a tube left lying in a vessel, and a leaning flask."""
    import math

    up = np.eye(3)
    beaker = np.array([0.30, 0.03, t.TOP])
    flask = np.array([0.30, -0.08, t.TOP])
    B, F = (beaker, up), (flask, up)
    over = beaker + np.array([0.0, 0.0, 0.06])
    beside = beaker + np.array([0.10, 0.0, 0.06])
    assert t.over_beaker(over, B) and not t.over_beaker(beside, B)
    print("ok  a slide release counts only with the bar over the beaker's opening")
    assert not t.tube_in_vessels(beaker + np.array([0.08, 0.0, 0.0]), up, F, B)
    assert t.tube_in_vessels(beaker + np.array([0.0, 0.0, 0.03]), up, F, B)
    lying = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])   # tube along -x, its middle over the beaker
    assert t.tube_in_vessels(beaker + np.array([0.15, 0.0, 0.05]), lying, F, B)
    print("ok  any part of the tube inside a vessel counts as in it (not only its tip)")
    assert t.tube_in_vessels(flask + np.array([0.0, 0.0, 0.17]), up, F, B)
    print("ok  a tube down the flask's neck counts as in it")
    # a flask held leaning: the neck's glass is where the flask really is
    side = np.array([0.0, 1.0, 0.0])
    for deg, axis_off, scraped in ((15.0, 0.0, True), (25.0, 0.0, True), (3.0, -t.BUNDLE, False), (8.0, -t.BUNDLE, False)):
        c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
        Rf = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])   # leaning about x
        tip_l = np.array([0.0, axis_off, sa.FLASK_H - 0.01])
        bar_l = tip_l + side * (sa.TUBE_R + sa.BAR_R + t.SIDE_GAP) + np.array([0.0, 0.0, sa.BAR_L / 2])
        got = t.touches_neck(flask + Rf @ bar_l, Rf @ t._R_SIDE, (flask, Rf))
        assert got == scraped, (deg, axis_off, got)
        print(f"ok  flask leaning {deg:.0f} deg, tube {'on its axis' if axis_off == 0.0 else 'centred with the bar'}: "
              f"scraped off = {got}")
    # a bar that leaves the beaker after the release
    assert not t.over_beaker(beaker + np.array([0.06, 0.0, 0.004]), B)
    print("ok  a bar out of the beaker after the release no longer counts as let go into it")


if __name__ == "__main__":
    more_routes()
