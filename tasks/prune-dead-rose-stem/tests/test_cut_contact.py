from types import SimpleNamespace
import math

import numpy as np
import pytest

import task
from cut_checks import footprint_distance, section_contact


@pytest.fixture(autouse=True)
def reset():
    task._clear()


def cutting_setup(monkeypatch):
    tool = SimpleNamespace(joint_names=["pivot"], data=SimpleNamespace(joint_pos=np.zeros((1, 1))))
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene={"secateurs": tool, "bush": object()}))
    base = task.sa.on_cane(0) + np.array([*task.sa.POT_XY, 0.0])
    target = ("dead", base, base + task.sa.TOP_S * task.sa.CANE_DIR, 0.0)
    root = task.sa.on_cane(.110) + np.array([*task.sa.POT_XY, 0.0]) - np.array([0, .03, task.sa.BLADE_Z])
    pose = {"o": root.copy(), "r": np.eye(3)}
    writes = []
    monkeypatch.setattr(task, "_np", lambda x: np.asarray(x))
    monkeypatch.setattr(task, "_body", lambda *args: (pose["o"], pose["r"]))
    monkeypatch.setattr(task, "_canes", lambda *args: [target])
    monkeypatch.setattr(task, "_write_joints", lambda asset, values: writes.append(dict(values)))
    monkeypatch.setattr(task, "_chain_gains", lambda *args: None)
    def observe(q, offset=(0, 0, 0), yaw=0):
        tool.data.joint_pos[0, 0] = q
        pose["o"] = root + np.asarray(offset)
        c, s = math.cos(yaw), math.sin(yaw)
        pose["r"] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        task._check_cut(env, np.zeros(3))
    return observe, writes


def test_continuous_partial_closure_then_withdrawal_cannot_invent_good_cut(monkeypatch):
    observe, writes = cutting_setup(monkeypatch)
    for q in np.linspace(0, .34, 25):
        observe(q, yaw=-q/2)
    for yaw in np.linspace(-.17, 0, 15):
        observe(.34, yaw=yaw)
    for t in np.linspace(0, 1, 20):
        observe(.34, (.03*t, .03*t, 0))
    side = np.array([.03, .03, 0.0])
    axial = (.068 - .110) * task.sa.CANE_DIR
    for t in np.linspace(0, 1, 25):
        observe(.34, side + t*axial)
    for q in np.linspace(.34, task.sa.CLOSE, 25):
        observe(q, side + axial)
    assert not task._S["cut"]
    assert task._S["pruning"].first_cut is None
    assert not task._S["cut_events"]
    assert not writes
    assert task.score({}) == 0


def test_actual_closed_blade_contact_survives_leaving_angular_gap(monkeypatch):
    observe, writes = cutting_setup(monkeypatch)
    offset = (.068 - .110) * task.sa.CANE_DIR
    for q in np.linspace(0, task.sa.CLOSE, 35):
        observe(q, offset, yaw=-q/2)
    assert not task._S["in_jaw"]["dead"]  # bypass blade has passed the angular gap
    assert task._S["cut"] and writes
    assert task._S["pruning"].first_cut == pytest.approx(.068)
    assert task._S["cut_events"][0]["contact"] == "blade_mesh"
    assert task._S["cut_events"][0]["blade_distance"] <= task.sa.R0


def square(cx, cy, half):
    return np.array([[cx-half, cy-half], [cx+half, cy-half], [cx+half, cy+half], [cx-half, cy+half]])


def test_thin_blade_crosses_section_even_when_all_section_vertices_miss():
    section = square(0, .03, .004)
    blade = np.array([[-.01, .029], [.01, .029], [.01, .031], [-.01, .031]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    assert all(footprint_distance(point, blade, faces) > 0 for point in section)
    assert section_contact(section, blade, faces, (.008, .055)) is not None


@pytest.mark.parametrize("section_half,blade_half", [(.01, .002), (.002, .01)])
def test_blade_and_bud_section_containment(section_half, blade_half):
    section = square(0, .03, section_half)
    blade = square(0, .03, blade_half)
    assert section_contact(section, blade, np.array([[0, 1, 2], [0, 2, 3]]), (.008, .055)) is not None


def test_finite_blade_near_miss_does_not_damage_section():
    section = square(.02, .03, .003)
    blade = square(0, .03, .002)
    assert section_contact(section, blade, np.array([[0, 1, 2], [0, 2, 3]]), (.008, .055)) is None


def test_early_real_bud_sweep_stays_damaged_after_closed_blade_has_passed(monkeypatch):
    tool = SimpleNamespace(joint_names=["pivot"], data=SimpleNamespace(joint_pos=np.zeros((1, 1))))
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene={"secateurs": tool, "bush": object()}))
    center = (task.BUD_VERTICES.min(axis=0) + task.BUD_VERTICES.max(axis=0)) / 2
    vertices = task.BUD_VERTICES - center + np.array([.008, .03, task.sa.BLADE_Z])
    targets = [(f"bud{k}", vertices[a], vertices[b], 0.) for k, (a, b) in enumerate(task.BUD_EDGES)]
    points = []
    for _, a, b, _ in targets:
        if abs(b[2] - a[2]) > 1e-10:
            t = (task.sa.BLADE_Z - a[2]) / (b[2] - a[2])
            if 0 <= t <= 1:
                points.append(a + t * (b - a))
    def overlap(q):
        return section_contact([task._blade_frame(x, q) for x in points],
                               task._BLADE_MESH.v, task._BLADE_MESH.f, task.sa.JAW_REACH)
    assert overlap(0) is None
    assert overlap(.17) is not None
    assert overlap(task.sa.CLOSE) is None
    writes = []
    monkeypatch.setattr(task, "_np", lambda x: np.asarray(x))
    monkeypatch.setattr(task, "_body", lambda *args: (np.zeros(3), np.eye(3)))
    monkeypatch.setattr(task, "_canes", lambda *args: targets)
    monkeypatch.setattr(task, "_write_joints", lambda asset, values: writes.append(dict(values)))
    for ix, q in enumerate([0, .17, .34, .51, task.sa.CLOSE]):
        task._S["t"] = ix
        tool.data.joint_pos[0, 0] = q
        task._check_cut(env, np.zeros(3))
        assert task._S["pruning"].bud_damaged == (ix > 0)
    assert len(task._S["cut_events"]) == 1
    event = task._S["cut_events"][0]
    assert event["q"] == .17 and event["t"] == .05
    assert event["contact"] == "blade_mesh_section"
    assert event["point"][2] == pytest.approx(task.sa.BLADE_Z)
    assert writes == [{"stub_bud_slide": task.sa.BUD_SINK}]


@pytest.mark.parametrize("height,expected_cut", [(.068, True), (.110, False), (.180, True)])
def test_contact_uses_visible_green_brown_and_top_radii(monkeypatch, height, expected_cut):
    observe, writes = cutting_setup(monkeypatch)
    q = task.sa.CLOSE
    angle = q - task.sa.HALF
    c, sn = math.cos(angle), math.sin(angle)
    vertices = task._BLADE_MESH.v[:, :2] @ np.array([[c, sn], [-sn, c]])
    # At the mesh's extreme X vertex this outward displacement is the exact
    # distance to every projected triangle, not an assumed circular blade.
    point = vertices[vertices[:, 0].argmax()] + np.array([(task.sa.R0 + task.sa.R1) / 2, 0])
    assert task.sa.JAW_REACH[0] < np.linalg.norm(point) < task.sa.JAW_REACH[1]
    distance = footprint_distance(task._blade_frame(point, q), task._BLADE_MESH.v, task._BLADE_MESH.f)
    assert distance == pytest.approx(.004925)
    offset = np.array([-point[0], .03 - point[1], 0.]) + (height - .110) * task.sa.CANE_DIR
    observe(0, offset)
    observe(q, offset)
    assert bool(writes) == expected_cut
    assert task._S["cut"] == expected_cut
    if expected_cut:
        assert task._S["pruning"].first_cut == pytest.approx(height)
