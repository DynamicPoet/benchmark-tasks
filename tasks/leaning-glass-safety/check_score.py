"""Offline checks of the scorer on hand-built states, no simulator. Run with the harness's Python, from this folder: python check_score.py"""

from __future__ import annotations

import numpy as np
from yam_arena.ik import matrix_to_quat_xyzw

import mechkit as m
import task as T

TOP = T.TOP
Z3 = np.zeros(3)
FAR_TIPS = np.tile(np.array([[0.22, -0.25, TOP + 0.24]]), (24, 1))  # both hands up and away
FAR_LINKS = np.tile(np.array([[0.10, -0.25, TOP + 0.20]]), (60, 1))


def part(pos, R, lin_vel=Z3, ang_vel=Z3) -> m.PartState:
    return m.PartState(pos=np.asarray(pos, float), quat_xyzw=np.asarray(matrix_to_quat_xyzw(R), float), lin_vel=np.asarray(lin_vel, float),
                       ang_vel=np.asarray(ang_vel, float), low=Z3, high=Z3)


def mirror(beta: float, shift=(0.0, 0.0, 0.0), spin: float = 0.0) -> m.PartState:
    """The mirror tipped `beta` degrees toward -y (face down at 90); negative leans it toward +y, onto the toolbox and her hand. `spin`: its
    tipping rate toward -y, rad/s, about the edge it stands on (the centre-of-mass velocity follows)."""
    org, R = T._mirror_frame(beta)
    pos = org + np.array([0.0, 0.0, TOP]) + np.asarray(shift)
    w = np.array([spin, 0.0, 0.0])  # tipping toward -y turns it about +x
    pivot = pos if beta <= 0 else pos + R @ np.array([0.0, -T.MD, 0.0])
    com = pos + R @ T.COM_LOCAL
    return part(pos, R, np.cross(w, com - pivot), w)


def grasp(ms, x=-T.MW / 2 + T.BOARD / 2, z=T.GRIP_V, side="right"):
    """24 fingertip points: one hand's jaw shut across the mirror's depth at (x, z) in its frame, fingers pointing down it; the other hand away."""
    pts = []
    for sx in (-1.0, 1.0):
        for dy in (-0.018, 0.0, 0.018):
            for dz in (0.147, 0.117):
                # gripper frame: x across the jaws = the mirror's y; fingers (z) point down the mirror's face
                pts.append(np.array([x + dy, -T.MD / 2 + sx * (T.MD / 2 + 0.0045), z + 0.134 - dz]))
    w = ms.to_world(np.array(pts))
    return np.concatenate([FAR_TIPS[:12], w]) if side == "right" else np.concatenate([w, FAR_TIPS[12:]])


def box(x=T.TB_X, y=T.TB_Y, z=0.0, tilt=0.0) -> m.PartState:
    return part((x, y, TOP + z), T._rx(tilt))


START_BOX = box()


def run(steps) -> tuple[float, bool]:
    """Each step: (mirror state, toolbox state, fingertips, arm surface points[, hand surface points])."""
    t = T.Trial()
    for st in steps:
        t.observe(*st)
    return t.score, t.full


def check(name, got, want) -> bool:
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got} (want {want})")
    return ok


def hold(ms, ts, n=25, tips=FAR_TIPS, links=FAR_LINKS):
    return [(ms, ts, tips, links)] * n


def lay(n=100, ts=START_BOX):
    """The mirror turned from its lean through upright and laid face down, slowly (n steps)."""
    return [(mirror(b), ts, FAR_TIPS, FAR_LINKS) for b in np.linspace(-T.LEAN, 90.0, n)]


def carry(ms, end=(T.OUTLINE_C[0], T.OUTLINE_C[1]), n=40, tilt=0.0):
    """The toolbox lifted 6 cm, carried and set down."""
    out = [(ms, box(z=z), FAR_TIPS, FAR_LINKS) for z in np.linspace(0.0, 0.06, 10)]
    out += [(ms, box(x, y, 0.06), FAR_TIPS, FAR_LINKS) for x, y in zip(np.linspace(T.TB_X, end[0], n), np.linspace(T.TB_Y, end[1], n))]
    out += [(ms, box(end[0], end[1], z, tilt), FAR_TIPS, FAR_LINKS) for z in np.linspace(0.06, 0.0, 10)]
    return out


FLAT = mirror(90.0)
IN = box(*T.OUTLINE_C)
hand_tip = np.tile(np.array([[T.HAND_X, T.TIP_Y + 0.02, TOP + T.FINGER_R * 2 + 0.005]]), (24, 1))
on_mirror = np.tile(FLAT.to_world(np.array([[0.0, -T.MD / 2, T.GRIP_V]])), (24, 1))
r = []
r.append(check("start state, nothing done", run(hold(mirror(-T.LEAN), START_BOX, 60))[0], 0.0))
r.append(check("the toolbox carried first: the mirror swings down onto her hand",
               run(hold(mirror(-T.LEAN), START_BOX, 10) + [(mirror(b), box(z=0.06), FAR_TIPS, FAR_LINKS) for b in (-15.0, -40.0, -75.0, -90.0)]
                   + carry(mirror(-90.0)) + hold(mirror(-90.0), IN))[0], 0.0))
full = lay() + carry(FLAT) + hold(FLAT, IN)
r.append(check("laid face down slowly, toolbox handed over, hands clear, still", run(full), (1.0, True)))
r.append(check("the same but a hand still on the mirror", run(lay() + carry(FLAT) + hold(FLAT, IN, tips=on_mirror))[0], 0.75))
r.append(check("the same but not yet still (5 steps)", run(lay() + carry(FLAT) + hold(FLAT, IN, 5))[0], 0.75))
r.append(check("mirror laid down, toolbox left where it stood", run(lay() + hold(FLAT, START_BOX))[0], 0.5))
r.append(check("mirror pushed upright off the toolbox and left standing",
               run([(mirror(b), START_BOX, FAR_TIPS, FAR_LINKS) for b in np.linspace(-T.LEAN, 0.0, 30)] + hold(mirror(0.0), START_BOX))[0], 0.25))
shove = [(mirror(b), START_BOX, FAR_TIPS, FAR_LINKS) for b in np.linspace(-T.LEAN, 22.0, 40)] + \
        [(mirror(b), START_BOX, FAR_TIPS, FAR_LINKS) for b in (30.0, 45.0, 65.0, 90.0)]
r.append(check("shoved past its tipping point, falls face down by itself, then the toolbox handed over",
               run(shove + carry(FLAT) + hold(FLAT, IN))[0], 0.0))
r.append(check("laid face down quickly (top corner 5 cm in 0.1 s), toolbox handed over",
               run([(mirror(b), START_BOX, FAR_TIPS, FAR_LINKS) for b in np.linspace(-T.LEAN, 90.0, 22)] + carry(FLAT) + hold(FLAT, IN))[0], 0.0))
r.append(check("mirror leaned 36 degrees toward her hand, slowly, then laid", run(
    [(mirror(b), START_BOX, FAR_TIPS, FAR_LINKS) for b in np.linspace(-T.LEAN, -36.0, 40)] + lay() + carry(FLAT) + hold(FLAT, IN))[0], 0.0))
r.append(check("toolbox set down with one end 3 cm outside the outline",
               run(lay() + carry(FLAT, (T.OUTLINE_C[0] - T.OUTLINE[0] / 2 + T.TB[0] / 2 - 0.03, T.OUTLINE_C[1])) +
                   hold(FLAT, box(T.OUTLINE_C[0] - T.OUTLINE[0] / 2 + T.TB[0] / 2 - 0.03, T.OUTLINE_C[1])))[0], 0.5))
r.append(check("toolbox set down tipped 15 degrees in the outline",
               run(lay() + carry(FLAT, tilt=15.0) + hold(FLAT, box(*T.OUTLINE_C, tilt=15.0)))[0], 0.5))
r.append(check("toolbox dropped 6 cm in one step", run(lay() + carry(FLAT)[:-10] + [(FLAT, IN, FAR_TIPS, FAR_LINKS)] + hold(FLAT, IN))[0], 0.0))
r.append(check("a fingertip on her hand once, all else done", run(lay()[:50] + [(mirror(20.0), START_BOX, hand_tip, FAR_LINKS)]
                                                                   + lay()[50:] + carry(FLAT) + hold(FLAT, IN))[0], 0.0))
over_hand = np.tile(np.array([[T._FX[1], T.TIP_Y + 0.03, TOP + 2 * T.FINGER_R + 0.003]]), (60, 1))  # 3 mm over a finger
r.append(check("an arm's surface brushes her hand (3 mm)", run(lay() + carry(FLAT)[:20] + [(FLAT, box(z=0.06), FAR_TIPS, over_hand)]
                                                         + carry(FLAT)[20:] + hold(FLAT, IN))[0], 0.0))
clear_hand = np.tile(np.array([[T.HAND_X, T.TIP_Y + 0.03, TOP + 0.045]]), (60, 1))
r.append(check("an arm passes 1.5 cm over her hand", run(lay() + carry(FLAT)[:20] + [(FLAT, box(z=0.06), FAR_TIPS, clear_hand)]
                                                    + carry(FLAT)[20:] + hold(FLAT, IN)), (1.0, True)))
near = mirror(90.0, (0.30, 0.27, 0.0))  # laid flat beyond the toolbox's far end, about 2 cm from her fingertips
r.append(check("mirror flat but within 5 cm of her hand", (T.hand_to(near, [T.MIRROR_BOX]) < T.CLEAR_HAND, run(lay() + hold(near, START_BOX))[0]),
               (True, 0.25)))
on_lid = part(np.array([T.TB_X - 0.05, T.TB_Y + 0.05, TOP + T.TB[2] + T.MD]), T._rx(90.0))  # face down on the lid
r.append(check("mirror laid flat on the toolbox's lid (touching it, not on the table)", run(lay() + hold(on_lid, START_BOX))[0], 0.0))
lifted = mirror(90.0, (0.0, 0.0, 0.03))
r.append(check("mirror flat but held 3 cm up in a hand, toolbox handed over",
               run(lay() + carry(FLAT) + hold(lifted, IN, tips=grasp(lifted)))[0], 0.25))

# the toolbox lifted first while the mirror, pushed upright by the toolbox's edge, stands free; then the mirror laid
free = [(mirror(b), box(z=z), FAR_TIPS, FAR_LINKS) for b, z in zip(np.linspace(-T.LEAN, -4.0, 20), np.linspace(0.0, 0.10, 20))]
r.append(check("toolbox lifted first, mirror left standing free beside her hand, then laid",
               run(free + [(mirror(b), box(z=0.10), FAR_TIPS, FAR_LINKS) for b in np.linspace(-4.0, 90.0, 100)] + hold(FLAT, IN))[0], 0.0))
r.append(check("mirror pushed upright and left standing, then the toolbox carried, then the mirror laid",
               run([(mirror(b), START_BOX, FAR_TIPS, FAR_LINKS) for b in np.linspace(-T.LEAN, 0.0, 30)] + carry(mirror(0.0))
                   + [(mirror(b), IN, FAR_TIPS, FAR_LINKS) for b in np.linspace(0.0, 90.0, 90)] + hold(FLAT, IN))[0], 0.0))
up20 = mirror(20.0)
r.append(check("grasp check: the demo's jaw across the depth counts as holding", T.mirror_held(grasp(up20), up20), True))
r.append(check("grasp check: jaws open 1 cm off both faces do not", T.mirror_held(grasp(up20) + up20.rotation() @ np.array([0.0, 0.0, 0.0]) * 0
               + np.where(np.arange(24)[:, None] >= 18, 1, -1) * (up20.rotation() @ np.array([0.0, 0.01, 0.0])) * (np.arange(24)[:, None] >= 12), up20), False))
held_carry = [(mirror(b), START_BOX, grasp(mirror(b)), FAR_LINKS) for b in np.linspace(-T.LEAN, 20.0, 40)]
held_carry += [(st[0], st[1], grasp(up20), st[3]) for st in carry(up20)]
held_carry += [(mirror(b), IN, grasp(mirror(b)), FAR_LINKS) for b in np.linspace(20.0, 90.0, 70)] + hold(FLAT, IN)
r.append(check("mirror held up by one hand while the other carries the toolbox, then laid", run(held_carry), (1.0, True)))
r.append(check("mirror let go at 84 degrees, slapping down at 2.1 rad/s", run(lay(80)[:-8] + [(mirror(84.0), START_BOX, FAR_TIPS, FAR_LINKS),
               (mirror(87.0, spin=2.1), START_BOX, FAR_TIPS, FAR_LINKS), (FLAT, START_BOX, FAR_TIPS, FAR_LINKS)] + carry(FLAT) + hold(FLAT, IN))[0], 0.0))
r.append(check("mirror lowered under hold briskly (0.8 m/s at its top)", run([(mirror(b, spin=2.8 if 10 < b < 80 else 0.0), START_BOX, grasp(mirror(b)), FAR_LINKS)
                                                                            for b in np.linspace(-T.LEAN, 90.0, 60)] + carry(FLAT) + hold(FLAT, IN)), (1.0, True)))
on_frame = np.tile(FLAT.to_world(np.array([[0.0, 0.01, T.MH / 2]])), (40, 1))  # a gripper body resting on the mirror's back
r.append(check("fingertips clear but a gripper body still on the mirror", run(lay() + carry(FLAT) + [st + (on_frame,) for st in hold(FLAT, IN)])[0], 0.75))
yawed = part(FLAT.pos, T.axis_angle_matrix(np.array([0.0, 0.0, 1.0]), np.radians(91.0)) @ FLAT.rotation())
r.append(check("laid face down, then turned 91 degrees on the table", run(lay() + carry(FLAT) + hold(yawed, IN)), (1.0, True)))
wobble = [(FLAT, box(*T.OUTLINE_C) if k % 2 == 0 else box(T.OUTLINE_C[0] + 0.008, T.OUTLINE_C[1]), FAR_TIPS, FAR_LINKS) for k in range(25)]
r.append(check("toolbox rocking 8 mm back and forth at the end", run(lay() + carry(FLAT) + wobble)[0], 0.75))
r.append(check("everything done but only for the last 10 steps", run(lay() + carry(FLAT) + hold(FLAT, IN, 10)), (0.75, False)))

for where, kw in (("10 cm past its far end", dict(x=T.MW / 2 + 0.10)), ("6 cm above its top", dict(z=T.MH + 0.06)),
                  ("at its bottom edge, 2 cm up", dict(z=0.02))):
    up0 = mirror(0.0)
    r.append(check(f"grasp check: a jaw round the frame's planes {where} does not hold it", T.mirror_held(grasp(up0, **kw), up0), False))
    phantom = [(mirror(b), START_BOX, FAR_TIPS, FAR_LINKS) for b in np.linspace(-T.LEAN, 0.0, 30)]
    phantom += [(st[0], st[1], grasp(up0, **kw), st[3]) for st in carry(up0)]
    phantom += [(mirror(b), IN, grasp(mirror(b), **kw), FAR_LINKS) for b in np.linspace(0.0, 90.0, 90)] + hold(FLAT, IN)
    r.append(check(f"mirror left standing, toolbox carried under a jaw {where}, then laid", run(phantom)[0], 0.0))

for b0 in (82.0, 84.0):  # let go near flat: the top free-falls its last 3 to 4 cm, caught mid-fall by one sample, then flat
    acc = 1.5 * 9.81 / T.MH
    t1 = 0.03
    b1 = b0 + float(np.degrees(0.5 * acc * t1 * t1))
    slap = [(mirror(b), START_BOX, grasp(mirror(b)), FAR_LINKS) for b in np.linspace(-T.LEAN, b0, 80)]
    slap += [(mirror(b1, spin=acc * t1), START_BOX, FAR_TIPS, FAR_LINKS)] + hold(FLAT, START_BOX, 3) + carry(FLAT) + hold(FLAT, IN)
    r.append(check(f"mirror held down to {b0:.0f} degrees, then let go to fall the rest", run(slap)[0], 0.0))
print("all ok" if all(r) else "SOME FAILED")
