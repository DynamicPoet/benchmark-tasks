"""The cutting is held as it parts, lies wholly over the bin's opening, and the secateurs lie on the table, not in the tub."""
from types import SimpleNamespace

import numpy as np
import pytest

import task
from motion_checks import clear_of_tub, cutting_held, ends_over_rim

R0 = task.sa.R0
BIN = np.array([.26, .37, 0.0])
# A cutting lying on the bin's floor, both ends well inside the opening (station frame, heights above the table).
IN_BIN = (BIN + [-.05, -.03, .01], BIN + [.06, .05, .01])
# Its middle over the opening, one end resting on the rim outside it.
ACROSS_RIM = (BIN + [-.02, 0., .05], BIN + [.15, 0., .125])
TOOL_ON_TABLE = np.tile([.20, -.18, .01], (8, 1))


def closed_grip(bottom, top, offset=.003):
    """A gripper closed on the cutting near its top end, both pad middles `offset` from its surface."""
    axis = (top - bottom) / np.linalg.norm(top - bottom)
    side = np.cross(axis, [0., 0., 1.])
    if np.linalg.norm(side) < 1e-6:  # an upright cutting
        side = np.cross(axis, [1., 0., 0.])
    side /= np.linalg.norm(side)
    centre = bottom + .8 * (top - bottom)
    mids = [centre + side * (R0 + offset), centre - side * (R0 + offset)]
    return np.array(mids + mids), 2 * R0 + .002


def open_far():
    p = np.array([[-.5, -.5, .5], [-.5, -.45, .5], [-.5, -.5, .52], [-.5, -.45, .52]])
    return p, .09


def test_cutting_held_needs_a_closed_gripper_with_both_pads_at_the_cutting():
    bottom, top = np.zeros(3), np.array([0., 0., .17])
    pts, gap = closed_grip(bottom, top)
    assert cutting_held({"left": (pts, gap), "right": open_far()}, bottom, top, R0, task.HOLD_REACH, task.HOLD_GAP_EXTRA)[0]
    # Opened beyond the cutting's width plus 6 mm: not held, though the pads are near it.
    assert not cutting_held({"left": (pts, 2 * R0 + .0061)}, bottom, top, R0, task.HOLD_REACH, task.HOLD_GAP_EXTRA)[0]
    # One pad more than 4 mm from the surface: not held.
    far = pts.copy()
    far[1] = far[0] + (far[1] - far[0]) * 3
    assert not cutting_held({"left": (far, gap)}, bottom, top, R0, task.HOLD_REACH, task.HOLD_GAP_EXTRA)[0]
    pts5, _ = closed_grip(bottom, top, offset=.0045)
    assert not cutting_held({"left": (pts5, gap)}, bottom, top, R0, task.HOLD_REACH, task.HOLD_GAP_EXTRA)[0]


def test_both_ends_must_be_over_the_opening():
    inner = task.sa.BIN_R_TOP - task.sa.BIN_WALL
    assert ends_over_rim(IN_BIN, BIN, np.eye(3), task.sa.BIN_H, inner)
    assert ends_over_rim([(ACROSS_RIM[0] + ACROSS_RIM[1]) / 2], BIN, np.eye(3), task.sa.BIN_H, inner)
    assert not ends_over_rim(ACROSS_RIM, BIN, np.eye(3), task.sa.BIN_H, inner)


def test_secateurs_in_the_tub_or_over_a_tipped_tub_are_not_on_the_table():
    r = task.sa.BIN_R_TOP + .005
    assert clear_of_tub(TOOL_ON_TABLE, BIN, np.eye(3), task.sa.BIN_H, r)
    in_tub = np.tile(BIN + [.03, .02, .01], (8, 1))
    assert not clear_of_tub(in_tub, BIN, np.eye(3), task.sa.BIN_H, r)
    one_handle_over = TOOL_ON_TABLE.copy()
    one_handle_over[3] = BIN + [-.12, 0., .01]
    assert not clear_of_tub(one_handle_over, BIN, np.eye(3), task.sa.BIN_H, r)
    on_its_side = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])  # the tub's axis now points along +y
    beside_tipped = np.tile(BIN + [0., .10, .01], (8, 1))
    assert clear_of_tub(beside_tipped, BIN, np.eye(3), task.sa.BIN_H, r) is False  # inside the upright rim circle
    assert not clear_of_tub(beside_tipped, BIN, on_its_side, task.sa.BIN_H, r)
    assert clear_of_tub(np.tile(BIN + [.20, .06, .01], (8, 1)), BIN, on_its_side, task.sa.BIN_H, r)


def measure_env(monkeypatch, cutting, pads, tool_points=TOOL_ON_TABLE, parts_at=1):
    """Run task._measure on a synthetic scene. The dead cane parts at control step `parts_at`."""
    task._clear()
    task._S["ready"] = True  # the reset hook has run
    h = task.sa.BUD_TOP_S + .009

    class Scene(dict):
        env_origins = np.zeros((1, 3))

    scene = Scene(bush=object(), secateurs=object(), bin=SimpleNamespace(data=SimpleNamespace(
        root_link_pose_w=np.array([[BIN[0], BIN[1], task.TOP, 0, 0, 0, 1.0]]))))
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene=scene))

    def check_cut(*args):
        if not task._S["cut"] and task._S["t"] >= parts_at:
            task._S["pruning"].cut(h)
            task._S.update(cut=True, h=h)
            task._S["cut_events"].append({"target": "dead", "height": h, "t": task._S["t"] / 20.0})

    monkeypatch.setattr(task, "_np", lambda x: np.asarray(x))
    monkeypatch.setattr(task, "_body", lambda asset, name, origin: (
        np.array([*task.sa.POT_XY, 0.0]) if name == "pot" else np.array([.20, -.18, .01]), np.eye(3)))
    monkeypatch.setattr(task, "_check_cut", check_cut)
    monkeypatch.setattr(task, "_retained_plant", lambda *args: (h, True))
    monkeypatch.setattr(task, "_pads", lambda *args: pads())
    monkeypatch.setattr(task, "_tool_points", lambda *args: tool_points)
    monkeypatch.setattr(task, "_cutting", lambda *args: cutting())
    monkeypatch.setattr(task, "_log", lambda *args: None)
    return env


def test_cutting_not_held_as_it_parts_is_a_fall_that_stays(monkeypatch):
    """One hand cuts with nothing holding the cutting: even if it drops straight into the bin, that is a fall."""
    env = measure_env(monkeypatch, lambda: IN_BIN, lambda: {"left": open_far(), "right": open_far()})
    for _ in range(40):
        task._measure(env)
    assert task._S["unheld_at_parting"] and task._S["fell"]
    assert task._S["held_at_parting"]["held"] is False
    assert task._S["cut_events"][-1]["held_at_parting"]["held"] is False
    assert task.score({}) == 0.0


def test_cutting_held_as_it_parts_then_released_in_the_bin_scores_full(monkeypatch):
    state = {"step": 0}

    def pads():
        state["step"] += 1
        if state["step"] <= 3:  # holding through the parting step, then let go over the bin
            return {"left": closed_grip(*IN_BIN), "right": open_far()}
        return {"left": open_far(), "right": open_far()}

    env = measure_env(monkeypatch, lambda: IN_BIN, pads, parts_at=2)
    for _ in range(40):
        task._measure(env)
    assert task._S["held_at_parting"]["held"] is True
    assert not task._S["fell"] and task._S["in_bin"] and task._S["tool_down"]
    assert task.score({}) == 1.0


def test_cutting_across_the_rim_is_not_in_the_bin(monkeypatch):
    state = {"step": 0}

    def pads():
        state["step"] += 1
        return {"left": closed_grip(*ACROSS_RIM) if state["step"] <= 3 else open_far(), "right": open_far()}

    env = measure_env(monkeypatch, lambda: ACROSS_RIM, pads, parts_at=2)
    for _ in range(40):
        task._measure(env)
    assert not task._S["fell"] and task._S["over_bin"] and not task._S["in_bin"]
    assert task.score({}) == 0.5


def test_secateurs_dropped_into_the_tub_do_not_earn_full_credit(monkeypatch):
    state = {"step": 0}

    def pads():
        state["step"] += 1
        return {"left": closed_grip(*IN_BIN) if state["step"] <= 3 else open_far(), "right": open_far()}

    in_tub = np.tile(BIN + [.03, .02, .01], (8, 1))
    env = measure_env(monkeypatch, lambda: IN_BIN, pads, tool_points=in_tub, parts_at=2)
    for _ in range(40):
        task._measure(env)
    assert task._S["in_bin"] and not task._S["tool_down"]
    assert task.score({}) == 0.75


@pytest.mark.parametrize("offset,held", [(.0035, True), (.0045, False)])
def test_parting_reads_the_grip_at_that_very_step(monkeypatch, offset, held):
    env = measure_env(monkeypatch, lambda: IN_BIN, lambda: {"left": closed_grip(*IN_BIN, offset=offset)}, parts_at=3)
    for _ in range(5):
        task._measure(env)
    assert task._S["held_at_parting"]["t"] == pytest.approx(3 / 20.0)
    assert task._S["held_at_parting"]["held"] is held
    assert task._S["fell"] is (not held)
