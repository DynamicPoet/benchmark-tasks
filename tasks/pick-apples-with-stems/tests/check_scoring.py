"""Attack the scorer offline with synthetic states (no simulator)."""
import importlib.util, sys
import numpy as np
from pathlib import Path
TASK = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, TASK)
spec = importlib.util.spec_from_file_location("task", TASK + "/task.py"); T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
sa = T.sa
R = T.R_APPLE
B = np.array([*T.BUCKET_XY, T.TOP])
PAD_C = B[2] + sa.PAD_TOP + R          # an apple's centre resting on the pad
HOME = {k: T.apple_home(k)[0] for k in sa.SPURS}

def fresh():
    T._clear()
    return T._S["apple"]

def run(seq, tipped=False, still=True, squeeze=None):
    """seq: list of steps; each step {k: (centre, touched, broken)}. squeeze: {k: finger-pad forces (N) seen while it
    was held}. Returns the scorer's score and success."""
    A = fresh()
    pts = None
    for step in seq:
        T._S["step"] += 1
        for k, (c, touched, broken) in step.items():
            st = A[k]
            if broken is not None and st["broken"] is None:
                st["broken"] = broken
            others = [step[o][0] for o in step if o != k and A[o]["broken"] is not None]
            T.track_apple(st, np.asarray(c, float), touched, others, B)
        pts = np.vstack([T.apple_points(np.asarray(step[k][0], float), np.eye(3)) for k in sa.SPURS])
        T._S["history"].append(pts); del T._S["history"][:-(T.STILL_STEPS + 1)]
        T._S["cache"] = None
    for k, values in (squeeze or {}).items():
        for v in values:
            T.track_squeeze(A[k], v)
    if not still:
        T._S["history"][0] = T._S["history"][0] + 0.01
        T._S["cache"] = None
    T._S["tipped"] = tipped
    sc = T.score(None); su = T.success(None)
    return (1.0 if su else sc), {k: (A[k]["dropped"], A[k]["in_bucket"], A[k]["broken"]) for k in sa.SPURS}

def slot(dx, dy, z=None):
    return np.array([B[0] + dx, B[1] + dy, PAD_C if z is None else z])

S_A, S_B = slot(-0.03, 0.042), slot(-0.03, -0.042)
TABLE_REST = np.array([0.48, -0.32, T.TOP + R])  # clear of both the pot and the bucket
cases = []
# helper to build: apple carried (touched) to a point, released at height h above rest, then resting
def placed(k, rest, release_gap, broken="seat"):
    up = rest + np.array([0, 0, release_gap])
    return [(up, True, broken)] * 3 + [(up, False, broken)] + [(rest, False, broken)] * 25
def onrest(k, broken="seat"):
    return [(HOME[k], False, None)] * 28
def combine(a_seq, b_seq):
    n = max(len(a_seq), len(b_seq))
    a_seq = a_seq + [a_seq[-1]] * (n - len(a_seq)); b_seq = b_seq + [b_seq[-1]] * (n - len(b_seq))
    return [{"a": a, "b": b} for a, b in zip(a_seq, b_seq)]

tests = [
    ("both laid in, clean, still", combine(placed("a", S_A, 0.01), placed("b", S_B, 0.01)), {}, 1.0),
    ("both laid in, one still held (on the pad)", combine(placed("a", S_A, 0.01), [(S_B + [0, 0, 0.0], True, "seat")] * 28), {}, 0.75),
    ("held above the pad inside the bucket", combine(placed("a", S_A, 0.01), [(S_B + [0, 0, 0.05], True, "seat")] * 28), {}, 0.5),
    ("both stem pulls", combine(placed("a", S_A, 0.01, "stem"), placed("b", S_B, 0.01, "stem")), {}, 0.5),
    ("one stem pull", combine(placed("a", S_A, 0.01, "stem"), placed("b", S_B, 0.01)), {}, 0.75),
    ("dropped from the rim (20 cm)", combine(placed("a", S_A, 0.20), placed("b", S_B, 0.01)), {}, 0.5),
    ("dropped from 5.5 cm", combine(placed("a", S_A, 0.055), placed("b", S_B, 0.01)), {}, 0.5),
    ("released 4.5 cm up", combine(placed("a", S_A, 0.045), placed("b", S_B, 0.01)), {}, 1.0),
    ("second laid on the first, 1 cm", combine(placed("a", S_A, 0.01), [(S_B, True, "seat")] * 0 + placed("b", S_A + [0.01, 0, 2 * R - 0.0005], 0.01)), {}, 1.0),
    ("second dropped on the first from 8 cm", combine(placed("a", S_A, 0.01), placed("b", S_A + [0.01, 0, 2 * R - 0.0005], 0.08)), {}, 0.5),
    ("nothing done", combine(onrest("a"), onrest("b")), {}, 0.0),
    ("one picked, one left on the tree", combine(placed("a", S_A, 0.01), onrest("b")), {}, 0.5),
    ("both in but pot tipped", combine(placed("a", S_A, 0.01), placed("b", S_B, 0.01)), {"tipped": True}, 0.0),
    ("both in, still moving", combine(placed("a", S_A, 0.01), placed("b", S_B, 0.01)), {"still": False}, 0.75),
    ("set on the table, outside", combine(placed("a", np.array([B[0]+0.24, B[1], T.TOP + R]), 0.01), placed("b", S_B, 0.01)), {}, 0.5),
    ("knocked off, fell to the table", combine([(HOME["a"], False, "seat")] + [(TABLE_REST, False, "seat")] * 27, placed("b", S_B, 0.01)), {}, 0.5),
    ("on the floor", combine([(np.array([0.3, -0.7, T.TOP - 0.7]), False, "seat")] * 28, placed("b", S_B, 0.01)), {}, 0.5),
    ("dropped, then put back in by hand", combine(placed("a", S_A, 0.2) + [(S_A + [0, 0, 0.05], True, "seat")] * 5 + placed("a", S_A, 0.01), placed("b", S_B, 0.01)), {}, 0.5),
    ("set on the rim, rolls off onto the table", combine([(np.array([B[0], B[1] - sa.BKT_R_TOP, B[2] + sa.BKT_H + R]), True, "seat")] * 3 + [(np.array([B[0], B[1] - sa.BKT_R_TOP, B[2] + sa.BKT_H + R]), False, "seat")] * 3 + [(np.array([B[0], B[1] - sa.BKT_R_TOP - 0.03 - 0.01 * i, B[2] + sa.BKT_H + R - 0.04 * i]), False, "seat") for i in range(1, 6)] + [(np.array([B[0], B[1] - 0.25, T.TOP + R]), False, "seat")] * 20, placed("b", S_B, 0.01)), {}, 0.5),
    ("set down gently on the soil, then into the bucket", combine([(np.array([0.37, 0.05, T.TOP + sa.SOIL_Z + R + 0.005]), True, "seat")] * 3 + [(np.array([0.37, 0.05, T.TOP + sa.SOIL_Z + R]), False, "seat")] * 3 + [(S_A + [0, 0, 0.1], True, "seat")] * 3 + placed("a", S_A, 0.01), placed("b", S_B, 0.01)), {}, 1.0),
    ("set down gently on the table, then into the bucket", combine([(TABLE_REST + [0, 0, .003], True, "seat")] * 3 + [(TABLE_REST, False, "seat")] * 3 + [(S_A + [0, 0, 0.1], True, "seat")] * 3 + placed("a", S_A, 0.01), placed("b", S_B, 0.01)), {}, 1.0),
    ("held at 18 N, both laid in", combine(placed("a", S_A, 0.01), placed("b", S_B, 0.01)), {"squeeze": {"a": [18.0], "b": [18.0]}}, 1.0),
    ("held at exactly 24 N", combine(placed("a", S_A, 0.01), placed("b", S_B, 0.01)), {"squeeze": {"a": [24.0]}}, 1.0),
    ("squeezed at 24.1 N once, then gently", combine(placed("a", S_A, 0.01), placed("b", S_B, 0.01)), {"squeeze": {"a": [12.0, 24.1, 10.0]}}, 0.5),
    ("first apple squeezed at 40 N", combine(placed("a", S_A, 0.01), placed("b", S_B, 0.01)), {"squeeze": {"b": [40.0]}}, 0.5),
    ("both squeezed", combine(placed("a", S_A, 0.01), placed("b", S_B, 0.01)), {"squeeze": {"a": [30.0], "b": [30.0]}}, 0.0),
    ("stem pull, squeezed", combine(placed("a", S_A, 0.01, "stem"), placed("b", S_B, 0.01)), {"squeeze": {"a": [30.0]}}, 0.5),
    ("squeezed while still on the tree", combine(onrest("a"), placed("b", S_B, 0.01)), {"squeeze": {"a": [30.0]}}, 0.5),
    ("laid in, later lifted out and dropped on the table", combine(placed("a", S_A, 0.01) + [(S_A + [0, 0, 0.3], True, "seat")] * 4 + [(S_A + [0, 0, 0.3], False, "seat")] + [(TABLE_REST, False, "seat")] * 25, placed("b", S_B, 0.01)), {}, 0.5),
]
bad = 0
for name, seq, kw, want in tests:
    got, info = run(seq, **kw)
    if name in ("knocked off, fell to the table", "laid in, later lifted out and dropped on the table"):
        assert info["a"][0], name
    ok = abs(got - want) < 1e-9
    bad += not ok
    print(f"{'ok ' if ok else 'BAD'} {got:4.2f} (want {want:4.2f})  {name}  {info}")
print("failures:", bad)

assert bad == 0, f"{bad} scorer attacks failed"

# A real lateral fall clears the other sphere at all times, although it is
# directly above it at release. The first actual contact is the pad.
dt = T.PHYSICS_DT
gravity = 9.81
height = .085
speed = .8
t_hit = np.sqrt(2 * height / gravity)
times = np.linspace(0, t_hit, 400)
assert np.min(np.hypot(speed * times, height - gravity * times**2 / 2)) > 2 * R + .007

def drop_to_pad(height):
    pad = B[2] + sa.PAD_TOP
    st = fresh()["b"]
    st["broken"] = "seat"
    T.track_flight(st, pad + height, True, None, dt)
    for tick in range(1, 180):
        low = pad + max(0., height - gravity * (tick * dt)**2 / 2)
        T.track_flight(st, low, False, ("bucket", pad) if low == pad else None, dt)
    assert np.isclose(st["drop_from"], height)
    return st["dropped"]

assert drop_to_pad(height)
assert drop_to_pad(.058)  # first 20 Hz free-frame clearance would be 45.74 mm
assert not drop_to_pad(.05)
assert not drop_to_pad(.045)

from yam_arena.urdf import axis_angle_matrix
axis = np.ones(3) / np.sqrt(3)
angle = 2 * np.arcsin(.0058 / (2 * R))
rot = axis_angle_matrix(axis, angle)
def moving_sample(rotation):
    return np.vstack([T.apple_points(S_A, np.eye(3)), T.apple_points(S_B, rotation)])
first, second = moving_sample(np.eye(3)), moving_sample(rot)
assert np.linalg.norm(second-first, axis=1).max() < .00474
assert np.isclose(T.sphere_spread([first, second]), .0058)
assert np.isclose(T.sphere_spread([first, first+.003]), .003*np.sqrt(3))

# A rotated bucket's pad and walls move with it, including near the wall.
tilt = axis_angle_matrix(np.array([1., 0., 0.]), np.deg2rad(60))
local = S_A - B
assert T.in_bucket(B + tilt @ local, [], B, tilt)
assert not T.in_bucket(B + local, [], B, tilt)

# The filled part ends at soil level and the separate lip sectors leave
# the inside of the pot open up to the rim.
import re
colliders = sa.tree_colliders("tree")
def hull_points(text):
    match = re.search(r"point3f\[\] points = \[(.*?)\]", text, re.S)
    return np.asarray([list(map(float, row.split(","))) for row in re.findall(r"\(([^)]+)\)", match[1])])
assert np.isclose(hull_points(colliders[0])[:, 2].max(), sa.SOIL_Z)
sectors = [hull_points(text) for text in colliders if '"c_rim_' in text]
assert len(sectors) == 24
assert all(np.hypot(p[:, 0], p[:, 1]).min() >= .139 and np.isclose(p[:, 2].max(), sa.POT_H) for p in sectors)
# Pad squeeze is the force along each pad's own closing axis; friction along the skin does not count.
assert np.isclose(T.pad_squeeze([[0., 18., 9.]], [[0., 1., 0.]]), 18.)
assert np.isclose(T.pad_squeeze([[0., -18., 9.], [3., 0., 0.]], [[0., 2., 0.], [0., 0., 1.]]), 18.)
assert T.pad_squeeze(np.zeros((0, 3)), np.zeros((0, 3))) == 0.
st = {}
T.track_squeeze(st, 24.0); assert not st["bruised"]
T.track_squeeze(st, 24.01); assert st["bruised"]
T.track_squeeze(st, 1.0); assert st["bruised"] and np.isclose(st["peak_squeeze"], 24.01)
print("Release height, landing surface, bucket pose, exact skin displacement, open soil and squeeze checks passed")
