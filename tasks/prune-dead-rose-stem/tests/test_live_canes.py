"""The live canes beside the dead one: closing the blades across any of them, from any approach, scores 0 for good."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

import task

sa = task.sa
LEAN = sa._lean()[0]


@pytest.fixture(autouse=True)
def reset():
    task._clear()


def rotation(normal, jaws) -> np.ndarray:
    """Tool frame columns: x, y (the jaws' direction when open), z (the blade plane's normal)."""
    n = np.asarray(normal, float)
    n /= np.linalg.norm(n)
    y = np.asarray(jaws, float) - np.dot(jaws, n) * n
    y /= np.linalg.norm(y)
    return np.column_stack([np.cross(y, n), y, n])


def scene(monkeypatch):
    """The real cut targets of the bush (pot at the station's origin), with the secateurs' pose set by each test."""
    tool = SimpleNamespace(joint_names=["pivot"], data=SimpleNamespace(joint_pos=np.zeros((1, 1))))
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene={"secateurs": tool, "bush": object()}))
    poses = {"pot": (np.zeros(3), np.eye(3)), "stub_bud": (sa.on_cane(sa.BUD_S), LEAN)}
    writes = []
    monkeypatch.setattr(task, "_np", lambda x: np.asarray(x))
    monkeypatch.setattr(task, "_body", lambda asset, name, origin: poses.get(name, (np.zeros(3), np.eye(3))))
    monkeypatch.setattr(task, "_write_joints", lambda asset, values: writes.append(dict(values)))
    monkeypatch.setattr(task, "_chain_gains", lambda *args: None)

    def close_at(point, rot, radius=0.03):
        """Put `point` in the open jaws, `radius` from the pivot on the jaws' middle line, then squeeze fully."""
        poses["hook_half"] = (np.asarray(point, float) - rot @ np.array([0.0, radius, sa.BLADE_Z]), rot)
        for q in (0.0, 0.3, sa.CLOSE):
            tool.data.joint_pos[0, 0] = q
            task._check_cut(env, np.zeros(3))

    return close_at, writes


def live_point(cane, s):
    return cane.at(s)


HORIZONTAL = [rotation((0, 0, 1), (math.cos(a), math.sin(a), 0)) for a in np.radians([0, 45, 90, 135, 180, 225, 270, 315])]


@pytest.mark.parametrize("k", range(len(sa.LIVE_CANES)))
@pytest.mark.parametrize("fraction", [0.12, 0.3, 0.6, 0.9])
@pytest.mark.parametrize("r", range(len(HORIZONTAL)))
def test_closing_across_any_live_cane_from_any_side_scores_zero(monkeypatch, k, fraction, r):
    cane = sa.LIVE_CANES[k]
    close_at, _ = scene(monkeypatch)
    close_at(live_point(cane, fraction * cane.length), HORIZONTAL[r])
    assert task._S["green_cut"]
    assert task.score({}) == 0.0


@pytest.mark.parametrize("k", range(len(sa.LIVE_CANES)))
def test_live_cane_nearly_in_the_blade_plane_is_still_cut(monkeypatch, k):
    """A tool tilted until the cane almost lies in the blade's plane: the old plane-crossing test skipped this case."""
    cane = sa.LIVE_CANES[k]
    d = cane.direction
    side = np.cross(d, [0.0, 0.0, 1.0])
    side /= np.linalg.norm(side)
    normal = np.cross(d, side) + 0.05 * d  # nearly perpendicular to the cane
    close_at, _ = scene(monkeypatch)
    close_at(live_point(cane, 0.5 * cane.length), rotation(normal, side))
    assert task._S["green_cut"]


@pytest.mark.parametrize("k", range(len(sa.LIVE_CANES)))
@pytest.mark.parametrize("radius", [0.012, 0.02, 0.04])
def test_live_cane_anywhere_in_the_jaws_counts(monkeypatch, k, radius):
    """Near the pivot or far out towards the tips: a live cane between the jaws when they close is cut."""
    cane = sa.LIVE_CANES[k]
    close_at, _ = scene(monkeypatch)
    close_at(live_point(cane, 0.35 * cane.length), HORIZONTAL[2], radius=radius)
    assert f"green{k}" in [e["target"] for e in task._S["cut_events"]]


def test_demo_cut_closes_on_the_dead_cane_only(monkeypatch):
    close_at, writes = scene(monkeypatch)
    close_at(sa.on_cane(task.CUT_AT), HORIZONTAL[2], radius=task.CUT_RADIUS)
    assert task._S["cut"] and not task._S["green_cut"]
    assert [e["target"] for e in task._S["cut_events"]] == ["dead"]


@pytest.mark.parametrize("yaw", np.radians(np.arange(0, 360, 30)))
def test_a_careful_cut_fits_between_the_neighbours_from_any_side(monkeypatch, yaw):
    """The dead cane's neighbours are close, but jaws centred on the dead cane above its bud miss them from every side."""
    close_at, _ = scene(monkeypatch)
    close_at(sa.on_cane(task.CUT_AT), rotation((0, 0, 1), (math.cos(yaw), math.sin(yaw), 0)), radius=task.CUT_RADIUS)
    assert task._S["cut"]
    assert not task._S["green_cut"]


def test_jaws_aimed_at_the_neighbour_beside_the_dead_cane_cut_it(monkeypatch):
    """The careless miss of the failure demo: the same height, the neighbour nearer the robot."""
    close_at, _ = scene(monkeypatch)
    target = task._live_cane_point(sa.LIVE_CANES[0], sa.on_cane(task.CUT_AT)[2]) - np.array([*sa.POT_XY, 0.0])
    close_at(target, HORIZONTAL[2], radius=task.CUT_RADIUS + 0.008)
    assert task._S["green_cut"] and not task._S["cut"]
    assert task.score({}) == 0.0


def test_neighbours_are_close_to_the_dead_cane_and_flank_it():
    """At the bud's height one live cane lies on each side of the dead cane, 4 to 6 cm away (centre to centre)."""
    bud = sa.on_cane(sa.BUD_S)
    near = []
    for cane in sa.LIVE_CANES:
        p = cane.at((bud[2] - sa.SOIL_Z) / cane.direction[2])
        offset = p[:2] - bud[:2]
        if 0.04 <= np.linalg.norm(offset) <= 0.06:
            near.append(np.sign(offset[0]))
    assert -1 in near and 1 in near


def test_open_jaws_beside_a_live_cane_are_harmless(monkeypatch):
    """Blades that close near, but not on, a live cane do not count."""
    cane = sa.LIVE_CANES[1]
    close_at, _ = scene(monkeypatch)
    p = live_point(cane, 0.5 * cane.length)
    # Jaws pointing away from the cane, the pivot 2 cm beyond its surface.
    away = np.cross(cane.direction, [0.0, 0.0, 1.0])
    away /= np.linalg.norm(away)
    rot = rotation((0, 0, 1), away)
    close_at(p + away * (0.03 + 0.02 + cane.radius), rot)
    assert "green1" not in [e["target"] for e in task._S["cut_events"]]
