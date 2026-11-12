from types import SimpleNamespace

import numpy as np
import pytest

import task


def good_state():
    task._clear()
    h = task.sa.BUD_TOP_S + .009
    task._S["pruning"].cut(h)
    task._S.update(ready=True, cut=True, h=h, stub_height=h, bud_attached=True,
                   in_bin=True, tool_down=True, clear=True,
                   history=[np.zeros(6) for _ in range(21)])
    return h


def test_last_frame_cannot_supply_one_second_hold_and_queries_do_not_advance_it():
    good_state()
    task._S["full_conditions"] = [False] * 21
    task._record_full_conditions()
    before = list(task._S["full_conditions"])
    for _ in range(100):
        assert task.score({}) == .75
    assert task._S["full_conditions"] == before
    for _ in range(19):
        task._record_full_conditions()
    assert task.score({}) == .75
    task._record_full_conditions()
    assert task.score({}) == 1
    task._S["clear"] = False  # current state is checked even without another sample
    assert task.score({}) == .75
    task._record_full_conditions()
    task._S["clear"] = True
    for _ in range(20):
        task._record_full_conditions()
    assert task.score({}) == .75
    task._record_full_conditions()
    assert task.score({}) == 1
    task._S["history"][0][0] = .02
    assert task.score({}) == .75


def test_measure_records_once_per_control_step_and_uses_same_motion_window(monkeypatch):
    h = good_state()
    class Scene(dict):
        env_origins = np.zeros((1, 3))
    scene = Scene(bush=object(), secateurs=object(), bin=SimpleNamespace(data=SimpleNamespace(
        root_link_pose_w=np.array([[.20, .37, task.TOP, 0, 0, 0, 1.0]]))))
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene=scene))
    monkeypatch.setattr(task, "_np", lambda x: np.asarray(x))
    monkeypatch.setattr(task, "_body", lambda asset, name, origin: (
        np.array([*task.sa.POT_XY, 0.0]) if name == "pot" else np.array([.20, -.18, .01]), np.eye(3)))
    monkeypatch.setattr(task, "_check_cut", lambda *args: None)
    monkeypatch.setattr(task, "_retained_plant", lambda *args: (h, True))
    pads = np.array([[-.5, -.5, .5], [-.5, -.5, .6]])
    monkeypatch.setattr(task, "_pads", lambda *args: {"left": (pads, .09), "right": (pads, .09)})
    monkeypatch.setattr(task, "_tool_points", lambda *args: np.tile([.20, -.18, .01], (8, 1)))
    monkeypatch.setattr(task, "_cutting", lambda *args: (np.array([.20, .37, .04]), np.array([.20, .37, .10])))
    monkeypatch.setattr(task, "_log", lambda *args: None)
    for ix in range(20):
        task._measure(env)
        assert len(task._S["full_conditions"]) == ix + 1
        assert task.score({}) == .75
    task._measure(env)
    assert len(task._S["full_conditions"]) == 21
    assert all(task._S["full_conditions"])
    assert task.score({}) == 1  # not an accidental two-second stillness hold
    count = task._S["t"]
    for _ in range(10):
        assert task.score({}) == 1
    assert task._S["t"] == count
