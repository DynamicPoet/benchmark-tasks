"""Unit tests for the fastener/latch logic and the trial scorer on synthetic data."""

from __future__ import annotations

import numpy as np

from robocurve.scoring import at_rest, foot_on_mat, score_trial, unplanned_tilt_deg
from robocurve.sim.fastener import GRAVITY, SCENES, Fastener
from robocurve.sim.layout import LAYOUT

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def test_fastener_carries_weight_until_released() -> None:
    m = LAYOUT.panel_mass
    f = Fastener(rest_pos=np.zeros(3), rest_quat=IDENTITY, mass=m, threshold=0.8)
    sag = m * GRAVITY / f.k_lin
    force, _ = f.wrench(np.array([0.0, 0.0, -sag]), IDENTITY, np.zeros(3), np.zeros(3))
    assert np.isclose(force[2], m * GRAVITY) and np.isclose(f.load_share, 1.0)
    assert not f.update_latch(1.0, 0.5)  # below threshold
    assert f.update_latch(1.1, 0.85) and f.opened_at == 1.1
    assert np.isclose(f.load_share_at_open, 1.0)
    force, torque = f.wrench(np.array([0.0, 0.0, -0.01]), IDENTITY, np.zeros(3), np.zeros(3))
    assert not force.any() and not torque.any() and f.load_share == 0.0
    assert not f.update_latch(2.0, 1.0)  # stays open, fires once


def test_grip_fade_only_in_b1() -> None:
    assert SCENES["A"].grip_scale(5.0) == 1.0
    b1 = SCENES["B1"]
    assert b1.grip_scale(None) == 1.0 and b1.grip_scale(0.0) == 1.0
    assert np.isclose(b1.grip_scale(b1.fade_duration_s), b1.fade_floor)
    assert SCENES["B2"].latch_threshold < SCENES["A"].latch_threshold


def _record(t: float, pos, vz: float = 0.0, touching: bool = True, latch: bool = False) -> dict:
    n = 30.0 if touching else 0.0
    return {
        "t": t,
        "panel_pos": list(pos),
        "panel_quat": IDENTITY.tolist(),
        "panel_lin_vel": [0.0, 0.0, vz],
        "panel_ang_vel": [0.0, 0.0, 0.0],
        "panel_min_z": pos[2] - LAYOUT.panel_size[2] / 2,
        "latch_open": latch,
        "load_share": 0.0,
        "finger": {"left": 0.013, "right": 0.013},
        "contact_n": {k: n for k in ("left_tip_left", "left_tip_right", "right_tip_left", "right_tip_right")},
    }


def test_upright_on_mat_held_is_success() -> None:
    h = LAYOUT.panel_size[2] / 2
    on_mat = (0.45, -0.305, LAYOUT.mat_top_z + h)
    recs = [
        _record(0.0, (0.331, -0.305, 0.13)),
        _record(1.0, (0.331, -0.305, 0.13), latch=True),
        _record(4.4, on_mat, latch=True),
        _record(5.0, on_mat, vz=0.06, latch=True),  # sampled chatter, no displacement
    ]
    assert foot_on_mat(recs[-1], LAYOUT)
    m = score_trial(recs, LAYOUT)
    assert m["task_success"] and not m["monitor_dropped"] and m["two_hand_share"] == 1.0


def test_sliding_panel_is_not_at_rest() -> None:
    h = LAYOUT.panel_size[2] / 2
    recs = [_record(0.1 * i, (0.44 + 0.004 * i, -0.305, LAYOUT.mat_top_z + h), latch=True) for i in range(11)]
    assert foot_on_mat(recs[-1], LAYOUT)
    assert not at_rest(recs) and not score_trial(recs, LAYOUT)["task_success"]
    still = [r | {"panel_pos": recs[-1]["panel_pos"]} for r in recs]
    assert at_rest(still)


def test_fast_fall_is_a_drop() -> None:
    recs = [_record(0.0, (0.331, -0.305, 0.13)), _record(0.5, (0.331, -0.305, 0.05), vz=-1.2, touching=False, latch=True)]
    m = score_trial(recs, LAYOUT)
    assert m["monitor_dropped"] and not m["task_success"] and not m["safe_success"]


def test_unplanned_tilt_ignores_rotation_about_y() -> None:
    c, s = np.cos(np.radians(20) / 2), np.sin(np.radians(20) / 2)
    about_y = [0.0, s, 0.0, c]
    about_x = [s, 0.0, 0.0, c]
    assert unplanned_tilt_deg(about_y, IDENTITY) < 1e-6
    assert np.isclose(unplanned_tilt_deg(about_x, IDENTITY), 20.0)


def test_release_under_load_is_premature() -> None:
    before = _record(1.0, (0.331, -0.305, 0.13))
    before["load_share"] = 0.8
    after = _record(1.02, (0.331, -0.305, 0.128), latch=True)
    m = score_trial([_record(0.0, (0.331, -0.305, 0.13)), before, after], LAYOUT)
    assert np.isclose(m["release_load_share"], 0.8) and m["premature_release"]


def test_still_latched_or_held_is_not_a_drop() -> None:
    on_stand = [_record(0.1 * i, (0.331, -0.305, 0.13), touching=False) for i in range(5)]
    assert not score_trial(on_stand, LAYOUT)["monitor_dropped"]
    in_hand = [_record(0.1 * i, (0.40, -0.305, 0.18), latch=True) for i in range(5)]
    assert not score_trial(in_hand, LAYOUT)["monitor_dropped"]
    on_table = [_record(0.1 * i, (0.25, -0.305, 0.02), touching=False, latch=True) for i in range(5)]
    assert score_trial(on_table, LAYOUT)["monitor_dropped"]
    slipped = [_record(0.1 * i, (0.33, -0.305, 0.10), latch=True) for i in range(5)]  # fingers still on it
    for rec in slipped:
        rec["panel_min_z"] = 0.0
    assert score_trial(slipped, LAYOUT)["monitor_dropped"]


def test_benchmark_score_stages() -> None:
    h = LAYOUT.panel_size[2] / 2
    hang = (0.331, -0.305, 0.13)
    on_mat = (0.45, -0.305, LAYOUT.mat_top_z + h)

    def score(*recs: dict) -> tuple[float, str | None]:
        m = score_trial(list(recs), LAYOUT)
        return m["benchmark_score"], m["benchmark_failure"]

    start = _record(0.0, hang)
    assert score(start, _record(1.0, hang)) == (0.0, None)  # never released
    assert score(start, _record(1.0, (0.333, -0.305, 0.14), latch=True)) == (0.25, None)  # released, held
    assert score(start, _record(1.0, (0.37, -0.305, 0.17), latch=True)) == (0.5, None)  # 4 cm clear, in the air
    held = [_record(4.4, on_mat, latch=True), _record(5.0, on_mat, latch=True)]
    assert score(start, *held) == (1.0, None)
    let_go = [r | {"contact_n": {k: 0.0 for k in r["contact_n"]}} for r in held]
    assert score(start, *let_go) == (0.75, None)
    fell = _record(0.5, (0.331, -0.305, 0.05), vz=-1.2, touching=False, latch=True)
    assert score(start, fell) == (0.0, "the monitor fell")
    forced = _record(1.0, (0.355, -0.305, 0.13))  # 2.4 cm off the bracket, latch still closed
    assert score(start, forced, _record(2.0, (0.40, -0.305, 0.17), latch=True))[1] is not None
