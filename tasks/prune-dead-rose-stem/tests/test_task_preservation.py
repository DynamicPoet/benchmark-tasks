from types import SimpleNamespace

import numpy as np
import pytest

import task


@pytest.fixture(autouse=True)
def reset_tracker():
    task._clear()


def complete_good_cut():
    height = task.sa.BUD_TOP_S + 0.009
    task._S["pruning"].cut(height)
    task._S.update(cut=True, h=height, stub_height=height, bud_attached=True, in_bin=True,
                   tool_down=True, clear=True, history=[np.zeros(6) for _ in range(21)], full_conditions=[True] * 21)
    return height


def test_score_uses_final_stub_and_latched_bud_damage():
    height = complete_good_cut()
    assert task.score({}) == 1.0
    task._S["stub_height"] = task.sa.BUD_TOP_S + 0.002
    assert task.score({}) == 0.25
    task._S.update(stub_height=height, bud_attached=False)
    task._S["pruning"].observe_bud(False)
    assert task.score({}) == 0.0
    task._S.update(bud_attached=True)
    assert task.score({}) == 0.0


def test_retained_geometry_reads_actual_body_poses(monkeypatch):
    height = task.sa.BUD_TOP_S + 0.009
    rotation = task.sa._lean()[0]
    poses = {"pot": (np.zeros(3), np.eye(3)),
             "stub_green": (task.sa.on_cane(height - task.sa.L_SG), rotation),
             "stub_brown": (task.sa.on_cane(height - 0.001 - task.sa.L_SB), rotation),
             "stub_bud": (task.sa.on_cane(task.sa.BUD_S), np.eye(3))}
    monkeypatch.setattr(task, "_body", lambda asset, name, origin: poses[name])
    measured, attached = task._retained_plant(object(), np.zeros(3))
    assert measured == pytest.approx(height)
    assert attached
    poses["stub_bud"] = (task.sa.on_cane(task.sa.BUD_S) + np.array([0.01, 0, 0]), np.eye(3))
    assert not task._retained_plant(object(), np.zeros(3))[1]


def blade_fixture(monkeypatch, targets):
    tool = SimpleNamespace(joint_names=["pivot"], data=SimpleNamespace(joint_pos=np.array([[0.0]])))
    bush = object()
    environment = SimpleNamespace(unwrapped=SimpleNamespace(scene={"secateurs": tool, "bush": bush}))
    writes = []
    monkeypatch.setattr(task, "_np", lambda value: np.asarray(value))
    monkeypatch.setattr(task, "_body", lambda asset, name, origin: (np.zeros(3), np.eye(3)))
    monkeypatch.setattr(task, "_canes", lambda asset, origin: targets)
    monkeypatch.setattr(task, "_write_joints", lambda asset, values: writes.append(dict(values)))
    monkeypatch.setattr(task, "_chain_gains", lambda *args: None)
    def close():
        tool.data.joint_pos[0, 0] = 0.0
        task._check_cut(environment, np.zeros(3))
        tool.data.joint_pos[0, 0] = task.sa.CLOSE
        task._check_cut(environment, np.zeros(3))
    return close, writes


def cane_at(height, x=0.0, y=0.03, name="stub"):
    base = np.array([x, y, task.sa.BLADE_Z - height])
    return (name, base, base + np.array([0, 0, 0.24]), 0.0)


def test_tool_recut_shortens_visible_stub_and_removes_bud(monkeypatch):
    first_height = complete_good_cut()
    cut = task.sa.BUD_TOP_S - 0.005
    close, writes = blade_fixture(monkeypatch, [cane_at(cut)])
    close()
    assert task._S["pruning"].cuts == 2
    assert task._S["pruning"].stub_length == pytest.approx(cut)
    assert task._S["pruning"].bud_damaged
    assert writes[-1]["stub_green_slide"] == pytest.approx(cut - task.sa.BROWN_S)
    assert writes[-1]["stub_bud_slide"] == task.sa.BUD_SINK
    assert set(writes[-1]) == {"stub_green_slide", "stub_brown_slide", "stub_bud_slide"}
    assert task._S["h"] == first_height
    assert task._S["pruning"].first_cut == first_height
    assert task.score({}) == 0.0


def test_retrim_does_not_change_first_cutting_geometry_or_history(monkeypatch):
    first_height = complete_good_cut()
    history = [point.copy() for point in task._S["history"]]
    cut = task.sa.BUD_TOP_S + 0.006
    close, writes = blade_fixture(monkeypatch, [cane_at(cut)])
    close()
    assert task._S["pruning"].stub_length == pytest.approx(cut)
    assert task._S["h"] == first_height
    assert all(name.startswith("stub_") for values in writes for name in values)
    np.testing.assert_array_equal(task._S["history"], history)


def test_side_bud_cut_does_not_need_to_cross_cane_axis(monkeypatch):
    complete_good_cut()
    monkeypatch.setattr(task, "_body", lambda *args: (np.zeros(3), np.eye(3)))
    targets = [item for item in task._canes(object(), np.zeros(3)) if item[0].startswith("bud")]
    targets.append(cane_at(task.sa.BUD_TOP_S, x=0.025))
    center = (task.BUD_VERTICES.min(axis=0) + task.BUD_VERTICES.max(axis=0)) / 2
    origin = center - np.array([0.0, 0.03, task.sa.BLADE_Z])
    close, writes = blade_fixture(monkeypatch, targets)
    monkeypatch.setattr(task, "_body", lambda *args: (origin, np.eye(3)))
    close()
    assert [event["target"] for event in task._S["cut_events"]] == ["bud"]
    assert writes[-1] == {"stub_bud_slide": task.sa.BUD_SINK}
    assert task._S["pruning"].bud_damaged
    assert task.score({}) == 0.0


def test_canes_still_include_retained_stub_and_bud_after_first_cut(monkeypatch):
    height = complete_good_cut()
    monkeypatch.setattr(task, "_body", lambda *args: (np.zeros(3), np.eye(3)))
    monkeypatch.setattr(task, "_retained_plant", lambda *args: (height, True))
    targets = task._canes(object(), np.zeros(3))
    names = [target[0] for target in targets]
    assert "stub" in names and "dead" not in names
    assert all(f"bud{i}" in names for i in range(4))
    assert all(f"green{i}" in names for i in range(4))


def mesh_blade_case(monkeypatch, normal, *, plane_offset=0.0, reach=0.03, across=0.0):
    complete_good_cut()
    monkeypatch.setattr(task, "_body", lambda *args: (np.zeros(3), np.eye(3)))
    targets = [item for item in task._canes(object(), np.zeros(3)) if item[0].startswith("bud")]
    n = np.asarray(normal, float)
    n /= np.linalg.norm(n)
    reference = np.array([1.0, 0.0, 0.0]) if abs(n[2]) > 0.8 else np.array([0.0, 0.0, 1.0])
    x = np.cross(reference, n)
    x /= np.linalg.norm(x)
    rotation = np.column_stack([x, np.cross(n, x), n])
    center = (task.BUD_VERTICES.min(axis=0) + task.BUD_VERTICES.max(axis=0)) / 2
    origin = center - rotation @ np.array([across, reach, task.sa.BLADE_Z]) + n * plane_offset
    close, writes = blade_fixture(monkeypatch, targets)
    monkeypatch.setattr(task, "_body", lambda *args: (origin, rotation))
    close()
    return writes


@pytest.mark.parametrize("normal", [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 0.01, 0.02]])
def test_actual_bud_mesh_is_cut_from_any_blade_direction(monkeypatch, normal):
    writes = mesh_blade_case(monkeypatch, normal)
    assert task._S["pruning"].bud_damaged
    assert any(event["target"].startswith("bud") for event in task._S["cut_events"])
    assert writes[-1] == {"stub_bud_slide": task.sa.BUD_SINK}
    assert task.score({}) == 0.0


@pytest.mark.parametrize("options", [{"plane_offset": 0.02}, {"reach": 0.07}, {"across": 0.04}])
def test_plane_or_jaws_missing_actual_bud_are_harmless(monkeypatch, options):
    writes = mesh_blade_case(monkeypatch, [1, 1, 1], **options)
    assert not writes
    assert not task._S["pruning"].bud_damaged
    assert not task._S["cut_events"]
    assert task.score({}) == 1.0
