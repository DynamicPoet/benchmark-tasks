"""The scorer on hand-built states, no simulator: doing nothing, the lazy routes and the full score.

Run with the harness's Python from this folder:  python check_score.py
"""

from __future__ import annotations

import math
import os

import numpy as np

import task as T  # noqa: E402

UP = np.eye(3)
FAR = np.array([[0.0, 0.6, 1.5]] * 24)          # every fingertip far away
NO_CAM = [(np.array([0.0, 0.6, 1.6]), np.eye(3))] * 2


def start() -> dict:
    return {o.name: (np.array(o.pos, float), T._rotation(o.rot_xyzw)) for o in T.VIALS}


def in_bin(xy, k: int = 0):
    return np.array([xy[0] - 0.03 + 0.03 * (k % 3), xy[1] - 0.02 + 0.02 * (k // 3), T.TOP + 0.004]), UP


def facing_cam(pos, R, dist=0.10, off_deg=0.0):
    """A camera dist from the printed side, looking at it, the side's normal off_deg from the line of sight."""
    c = pos + R @ np.array([T.sa.LABEL_R, 0.0, T.LABEL_MID])
    n = R[:, 0]
    a = math.radians(off_deg)
    d = R @ np.array([math.cos(a), math.sin(a), 0.0])
    cam = c + dist * d
    back = d                                   # the camera looks along -back, at the label
    right = np.cross(np.array([0.0, 0.0, 1.0]), back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    return [(cam, np.column_stack([right, up, back])), NO_CAM[1]]


def run(steps) -> float:
    T._clear()
    S = T._S
    for st in steps:
        vials, tips, cams = st[:3]
        fingers = st[3] if len(st) > 3 else ()
        T.track(S, vials, tips, cams, fingers)
    return T.evaluate(S)


def hold(v, n=30):
    return [(v, FAR, NO_CAM)] * n


def grasp(pos, R):
    """The left hand closed on the vial's body from above, the right parked."""
    return T.gk.tips(left=T.gk.hand(pos + R @ np.array([0.0, 0.0, 0.03]), opening=T.sa.VIAL_R))


def blocking_finger(pos, R, cams):
    """A finger box halfway between the camera and the label's bold line."""
    p = pos + R @ np.array([T.sa.LABEL_R, 0.0, T.BOLD_Z])
    mid = 0.5 * (p + cams[0][0])
    return [(mid, np.eye(3), np.array([0.006, 0.006, 0.006]))]


def route(show: bool, inj_to: str | None, show_dist=0.10, show_off=0.0, flush_all=False, occlude=False,
          end_hover=None, held_in_flush=0, late_drop=False, end_held=False, wobble=False) -> float:
    v = start()
    steps = hold(v, 5)
    k = 0
    for name in T.DELIVERED:
        kind = T.KIND[name]
        pos, R = v[name]
        if show:
            # held up: the vial's printed side turned to a camera for 0.6 s (a finger on it)
            up = (pos + np.array([0.0, 0.0, 0.15]), R)
            v = {**v, name: up}
            touch = grasp(*up)
            cams = facing_cam(*up, dist=show_dist, off_deg=show_off)
            fingers = blocking_finger(*up, cams) if occlude else []
            steps += [(v, touch, cams, fingers)] * 12
        if kind == "flush" or flush_all:
            v = {**v, name: in_bin(T.FLUSH_XY, k)}
            k += 1
        elif inj_to == "returns":
            if held_in_flush:   # held (grasped) inside the flush bin for a while first
                inside = in_bin(T.FLUSH_XY, 5)
                steps += [({**v, name: inside}, grasp(*inside), NO_CAM)] * held_in_flush
            v = {**v, name: in_bin(T.RETURNS_XY)}
        elif inj_to == "on flush bin":
            v = {**v, name: (np.array([T.FLUSH_XY[0], T.FLUSH_XY[1], T.TOP + T.sa.BIN_H + 0.002]), T._rotation((0.0, 0.7071068, 0.0, 0.7071068)))}
        else:
            v = {**v, name: (pos, R)}
        steps += hold(v, 12)
    if late_drop:   # the Injection vial let go over the flush bin in the run's last 5 steps
        inj = [n for n in T.DELIVERED if T.KIND[n] == "injection"][0]
        inside = in_bin(T.FLUSH_XY, 5)
        steps += [({**v, inj: inside}, FAR, NO_CAM)] * 5
        return run(steps)
    if end_held:   # the run ends with the Injection vial grasped inside the flush bin
        inj = [n for n in T.DELIVERED if T.KIND[n] == "injection"][0]
        inside = in_bin(T.FLUSH_XY, 5)
        steps += [({**v, inj: inside}, grasp(*inside), NO_CAM)] * 30
        return run(steps)
    if wobble:   # a stocked vial moves 8 mm and comes back within the last second
        v1 = dict(v)
        p0, R0 = v["v1"]
        steps += hold(v, 25)
        steps += [({**v1, "v1": (p0 + np.array([0.008, 0.0, 0.0]), R0)}, FAR, NO_CAM)] * 3 + hold(v, 10)
        return run(steps)
    if end_hover is not None:   # one open fingertip held over the Injection vial's cap to the end
        inj = [n for n in T.DELIVERED if T.KIND[n] == "injection"][0]
        pos, R = v[inj]
        tip = pos + R @ np.array([0.0, 0.0, T.sa.VIAL_H + end_hover])
        pts = FAR.copy()
        pts[0] = tip
        steps += [(v, pts, NO_CAM)] * 60
        return run(steps)
    steps += hold(v, 30)
    return run(steps)


def main() -> None:
    cases = {
        "nothing": (run(hold(start(), 60)), 0.0),
        "every vial into the flush bin, never shown": (route(False, None, flush_all=True), 0.0),
        "every vial into the flush bin, read": (route(True, None, flush_all=True), 0.0),
        "three flush vials in, never shown, Injection left on the tray": (route(False, "tray"), 0.0),
        "read, three flush vials in, Injection left on the tray": (route(True, "tray"), 0.75),
        "read, three in, Injection to the returns bin": (route(True, "returns"), 1.0),
        "read only from 20 cm": (route(True, "returns", show_dist=0.20), 0.0),
        "read at 65 degrees off the printed side": (route(True, "returns", show_off=65.0), 0.0),
        "label held up but a finger covers it": (route(True, "returns", occlude=True), 0.0),
        "read, three in, Injection laid across the flush bin's rim": (route(True, "on flush bin"), 0.0),
        "all four in, a fingertip kept 15 mm over the Injection vial's cap": (route(True, None, flush_all=True, end_hover=0.015), 0.0),
        "all four in, a fingertip kept 5 mm over the Injection vial's cap": (route(True, None, flush_all=True, end_hover=0.005), 0.0),
        "Injection held inside the flush bin for 3 s, then to the returns bin": (route(True, "returns", held_in_flush=60), 1.0),
        "read from 10 cm at 55 degrees off the printed side": (route(True, "returns", show_off=55.0), 1.0),
        "read, three in, the run ends with the Injection vial held inside the flush bin": (route(True, "returns", end_held=True), 0.0),
        "full route, but a stocked vial moves 8 mm and back in the last second": (route(True, "returns", wobble=True), 0.75),
        "read, three in, Injection let go over the flush bin in the last 5 steps": (route(True, "returns", late_drop=True), 0.0),
    }
    os.environ.pop("TASK_SLOT", None)
    spots = {T.injection_spot(seed) for seed in range(200)}
    print(f"{'ok  ' if spots == {1, 2, 3, 4} else 'FAIL'} seeds 0..199 draw spots {sorted(spots)}")
    bad = 0
    for name, (got, want) in cases.items():
        ok = abs(got - want) < 1e-9
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {got:.2f} (want {want:.2f})")
    print("all pass" if not bad else f"{bad} failed")


if __name__ == "__main__":
    main()
