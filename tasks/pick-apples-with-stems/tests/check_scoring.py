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

def run(seq, tipped=False, still=True):
    """seq: list of steps; each step {k: (centre, touched, broken)}. Returns the scorer's score and success."""
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
    if not still:
        T._S["history"][0] = T._S["history"][0] + 0.01
        T._S["cache"] = None
    T._S["tipped"] = tipped
    sc = T.score(None); su = T.success(None)
    return (1.0 if su else sc), {k: (A[k]["dropped"], A[k]["in_bucket"], A[k]["broken"]) for k in sa.SPURS}

def slot(dx, dy, z=None):
    return np.array([B[0] + dx, B[1] + dy, PAD_C if z is None else z])

S_A, S_B = slot(-0.03, 0.042), slot(-0.03, -0.042)
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
    ("knocked off, fell to the table", combine([(HOME["a"], False, "seat")] + [(np.array([0.4, -0.05, T.TOP + R]), False, "seat")] * 27, placed("b", S_B, 0.01)), {}, 0.5),
    ("on the floor", combine([(np.array([0.3, -0.7, T.TOP - 0.7]), False, "seat")] * 28, placed("b", S_B, 0.01)), {}, 0.5),
    ("dropped, then put back in by hand", combine(placed("a", S_A, 0.2) + [(S_A + [0, 0, 0.05], True, "seat")] * 5 + placed("a", S_A, 0.01), placed("b", S_B, 0.01)), {}, 0.5),
    ("set on the rim, rolls off onto the table", combine([(np.array([B[0], B[1] - sa.BKT_R_TOP, B[2] + sa.BKT_H + R]), True, "seat")] * 3 + [(np.array([B[0], B[1] - sa.BKT_R_TOP, B[2] + sa.BKT_H + R]), False, "seat")] * 3 + [(np.array([B[0], B[1] - sa.BKT_R_TOP - 0.03 - 0.01 * i, B[2] + sa.BKT_H + R - 0.04 * i]), False, "seat") for i in range(1, 6)] + [(np.array([B[0], B[1] - 0.25, T.TOP + R]), False, "seat")] * 20, placed("b", S_B, 0.01)), {}, 0.5),
    ("set down gently on the soil, then into the bucket", combine([(np.array([0.37, 0.05, T.TOP + sa.SOIL_Z + R + 0.005]), True, "seat")] * 3 + [(np.array([0.37, 0.05, T.TOP + sa.SOIL_Z + R]), False, "seat")] * 3 + [(S_A + [0, 0, 0.1], True, "seat")] * 3 + placed("a", S_A, 0.01), placed("b", S_B, 0.01)), {}, 1.0),
    ("set down gently on the table, then into the bucket", combine([(np.array([0.3, -0.42, T.TOP + R + 0.003]), True, "seat")] * 3 + [(np.array([0.3, -0.42, T.TOP + R]), False, "seat")] * 3 + [(S_A + [0, 0, 0.1], True, "seat")] * 3 + placed("a", S_A, 0.01), placed("b", S_B, 0.01)), {}, 1.0),
    ("laid in, later lifted out and dropped on the table", combine(placed("a", S_A, 0.01) + [(S_A + [0, 0, 0.3], True, "seat")] * 4 + [(S_A + [0, 0, 0.3], False, "seat")] + [(np.array([0.3, -0.42, T.TOP + R]), False, "seat")] * 25, placed("b", S_B, 0.01)), {}, 0.5),
]
bad = 0
for name, seq, kw, want in tests:
    got, info = run(seq, **kw)
    ok = abs(got - want) < 1e-9
    bad += not ok
    print(f"{'ok ' if ok else 'BAD'} {got:4.2f} (want {want:4.2f})  {name}  {info}")
print("failures:", bad)

assert bad == 0, f"{bad} scorer attacks failed"
