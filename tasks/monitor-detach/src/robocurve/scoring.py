"""Per-trial metrics for the monitor-detach task, computed from per-step telemetry records.

Each record is a dict written by the simulation (robocurve.arena.monitor_detach on yam_arena); the
policy never sees these fields.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from robocurve.sim.fastener import quat_xyzw_to_matrix
from robocurve.sim.layout import Layout

Vec = npt.NDArray[np.float64]
Record = dict[str, Any]

DROP_SPEED = 0.5  # downward speed that counts as a drop [m/s]
CONTACT_N = 1.0  # fingertip force that counts as touching the panel [N]
UPRIGHT_TOL_DEG = 15.0
UNPLANNED_TOL_DEG = 15.0
HANDS_OPEN_TRAVEL = 0.03  # finger travel above which a gripper counts as open (0 = closed) [m]
# "At rest" is judged from displacement over a window, not from the sampled velocity: with both
# hands squeezing it, the panel chatters (0.01-0.06 m/s sampled) while staying put to 0.1 mm. The
# window is the benchmark's usual 1 s stillness test; 5 mm tolerates the ~2 mm sway of a monitor held
# by i2rt's soft arms on yam_arena (the scripted demos' last second moves 0.2-1.9 mm).
REST_WINDOW_S = 1.0
REST_MOVE_M = 0.005
REST_TURN_DEG = 3.0
GOALS = ("screen_up", "upright_held")
HELD_CLEARANCE = 0.02  # a panel whose lowest point is below this is resting on the table or plate [m]
# Benchmark score (the submission's task.json "scoring"): the highest stage reached, or 0 on a failure.
# Each condition can be judged on the real station from the trial video and a tape measure.
FORWARD_CLEAR_M = 0.03  # centre this far in front of where it hung (+x): clear of the stand
FORCED_MOVE_M = 0.01  # moved this far from its hanging pose while still latched: pulled off by force
STAGES = (
    (0.25, "released without falling"),
    (0.5, "carried clear of the stand"),
    (0.75, "standing upright on the mat"),
    (1.0, "standing upright on the mat, held by both grippers"),
)


def screen_up_deg(quat_xyzw: Sequence[float]) -> float:
    """Angle between the screen normal (panel +x) and world +z."""
    normal = quat_xyzw_to_matrix(np.asarray(quat_xyzw, dtype=float))[:, 0]
    return float(np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0))))


def unplanned_tilt_deg(quat_xyzw: Sequence[float], rest_quat_xyzw: Sequence[float]) -> float:
    """Rotation from the rest pose about any axis other than world y (the planned tilt axis).

    Swing-twist split of the relative rotation with twist axis y; returns the swing angle.
    """
    r_rel = quat_xyzw_to_matrix(np.asarray(quat_xyzw)) @ quat_xyzw_to_matrix(np.asarray(rest_quat_xyzw)).T
    # The swing is the rotation that takes y to r_rel @ y.
    y_new = r_rel[:, 1]
    return float(np.degrees(np.arccos(np.clip(y_new[1], -1.0, 1.0))))


def _touching(rec: Record, side: str) -> bool:
    """Both fingertips of ``side`` press on the panel: from contact forces, or from a record's ``holding``
    (the yam_arena port judges it from the fingers' gap and the tool point's place on the panel)."""
    if "holding" in rec:
        return bool(rec["holding"][side])
    forces = rec["contact_n"]
    return forces[f"{side}_tip_left"] > CONTACT_N and forces[f"{side}_tip_right"] > CONTACT_N


def foot_on_mat(rec: Record, layout: Layout) -> bool:
    """The panel's lowest point lies inside the mat's footprint, at mat height."""
    (mx0, my0, _), (mx1, my1, mz1) = layout.mat_box
    half = np.array(layout.panel_size) / 2
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
    world = corners @ quat_xyzw_to_matrix(np.asarray(rec["panel_quat"])).T + np.asarray(rec["panel_pos"])
    low = world[world[:, 2] < world[:, 2].min() + 0.01]
    x, y = low[:, 0].mean(), low[:, 1].mean()
    return bool(mx0 <= x <= mx1 and my0 <= y <= my1 and rec["panel_min_z"] < mz1 + 0.006)


def at_rest(records: Sequence[Record]) -> bool:
    """The panel moved less than REST_MOVE_M and turned less than REST_TURN_DEG over the last REST_WINDOW_S."""
    t_end = records[-1]["t"]
    if records[0]["t"] > t_end - REST_WINDOW_S + 1e-6:
        return False  # the trial is shorter than the window
    p_end = np.asarray(records[-1]["panel_pos"])
    r_end = quat_xyzw_to_matrix(np.asarray(records[-1]["panel_quat"]))
    for rec in reversed(records):
        if rec["t"] < t_end - REST_WINDOW_S - 1e-6:
            break
        r_rel = quat_xyzw_to_matrix(np.asarray(rec["panel_quat"])) @ r_end.T
        turn = np.degrees(np.arccos(np.clip((np.trace(r_rel) - 1.0) / 2.0, -1.0, 1.0)))
        if np.linalg.norm(np.asarray(rec["panel_pos"]) - p_end) > REST_MOVE_M or turn > REST_TURN_DEG:
            return False
    return True


def goal_reached(
    records: Sequence[Record],
    layout: Layout,
    arm_a: str = "left",
    arm_b: str = "right",
    goal: str = "upright_held",
) -> bool:
    """Whether the trial's latest record meets ``goal`` (see :func:`score_trial`)."""
    assert goal in GOALS, goal
    final = records[-1]
    if not (foot_on_mat(final, layout) and at_rest(records)):
        return False
    up = screen_up_deg(final["panel_quat"])
    if goal == "screen_up":
        hands_open = all(final["finger"][s] > HANDS_OPEN_TRAVEL for s in (arm_a, arm_b))
        return bool(up <= UPRIGHT_TOL_DEG and hands_open)
    held = _touching(final, arm_a) and _touching(final, arm_b)
    return bool(abs(up - 90.0) <= UPRIGHT_TOL_DEG and held)


def _upright_on_mat(records: Sequence[Record], layout: Layout) -> bool:
    final = records[-1]
    upright = abs(screen_up_deg(final["panel_quat"]) - 90.0) <= UPRIGHT_TOL_DEG
    return bool(upright and foot_on_mat(final, layout) and at_rest(records))


def benchmark_score(
    records: Sequence[Record], layout: Layout, *, dropped: bool, upright_on_mat: bool
) -> tuple[float, str, str | None]:
    """The submission's 0-1 score up to 0.75, as (score, stage, failure); score_trial adds the 1.0 stage.

    0 on a failure: the monitor fell (``dropped``), or it moved more than FORCED_MOVE_M off its hanging
    pose while the latch was still closed (pulled off by force). Otherwise the highest stage reached:
    0.25 the latch released (and, with no failure, the monitor never fell); 0.5 its centre at some point
    FORWARD_CLEAR_M in front of where it hung; 0.75 at rest at the end, standing on the mat within
    UPRIGHT_TOL_DEG of vertical. Heights are not used: held by one hand, the monitor hangs tilted in
    its plane, so its lowest corner says little about whether it is off the bracket.
    """
    start = records[0]
    x0 = start["panel_pos"][0]
    latched = [r for r in records if not r["latch_open"]]
    forced = max(np.linalg.norm(np.subtract(r["panel_pos"], start["panel_pos"])) for r in latched) if latched else 0.0
    if dropped:
        return 0.0, "none", "the monitor fell"
    if forced > FORCED_MOVE_M:
        return 0.0, "none", "pulled off the bracket without releasing the latch"
    released = [r for r in records if r["latch_open"]]
    cleared = [r for r in released if r["panel_pos"][0] >= x0 + FORWARD_CLEAR_M]
    if upright_on_mat and released:
        return (*STAGES[2], None)
    if cleared:
        return (*STAGES[1], None)
    if released:
        return (*STAGES[0], None)
    return 0.0, "none", None


def is_dropped(records: Sequence[Record], layout: Layout, arm_a: str = "left", arm_b: str = "right") -> bool:
    """A fast fall at any record, or ending off the mat and not held: still latched, or in a hand and clear
    of the table, is not. (A short fall onto the stand's plate can stay under DROP_SPEED.)"""
    final = records[-1]
    down = max(max(-r["panel_lin_vel"][2], r.get("peak_down_speed", 0.0)) for r in records)
    held = (_touching(final, arm_a) or _touching(final, arm_b)) and final["panel_min_z"] > HELD_CLEARANCE
    return bool(down > DROP_SPEED) or not (foot_on_mat(final, layout) or not final["latch_open"] or held)


def trial_score(
    records: Sequence[Record], layout: Layout, arm_a: str = "left", arm_b: str = "right"
) -> tuple[float, str, str | None]:
    """The benchmark score of a trial so far, as (score, stage, failure); :func:`score_trial` reports the
    same numbers among its metrics."""
    score, stage, failure = benchmark_score(
        records, layout, dropped=is_dropped(records, layout, arm_a, arm_b), upright_on_mat=_upright_on_mat(records, layout)
    )
    if score == 0.75 and goal_reached(records, layout, arm_a, arm_b):
        score, stage = STAGES[3]
    return score, stage, failure


def score_trial(
    records: Sequence[Record],
    layout: Layout,
    arm_a: str = "left",
    arm_b: str = "right",
    goal: str = "upright_held",
) -> dict[str, Any]:
    """Compute the per-trial metrics from telemetry records (in time order).

    ``goal`` sets what ``task_success`` checks at the end:
    - ``"screen_up"`` (the original design): at rest on the mat, screen normal within 15 deg of up,
      hands open.
    - ``"upright_held"`` (the task's goal): standing at rest on the mat, screen within
      15 deg of horizontal, held by both hands.
    """
    assert records, "no telemetry"
    assert goal in GOALS, goal
    rest_quat = records[0]["panel_quat"]
    t = np.array([r["t"] for r in records])
    # Downward speed per record; a record may also carry the peak over its control period.
    down = np.array([max(-r["panel_lin_vel"][2], r.get("peak_down_speed", 0.0)) for r in records])
    final = records[-1]
    on_mat = foot_on_mat(final, layout)
    # A drop is a fast fall, or ending off the mat and not held: still latched, or in a hand and clear
    # of the table, is not. (A short fall onto the stand's plate can stay under DROP_SPEED.)
    held = (_touching(final, arm_a) or _touching(final, arm_b)) and final["panel_min_z"] > HELD_CLEARANCE
    dropped = bool(np.max(down) > DROP_SPEED) or not (on_mat or not final["latch_open"] or held)

    task_success = goal_reached(records, layout, arm_a, arm_b, goal)

    i_open = next((i for i, r in enumerate(records) if r["latch_open"]), None)
    # The last record before the latch opened: once it is open the fastener carries nothing.
    release_load_share = None if i_open is None else float(records[max(i_open - 1, 0)]["load_share"])
    premature = False
    handoff_gap = None
    two_hand_share = None
    if i_open is not None:
        a_holding = _touching(records[i_open], arm_a)
        premature = (release_load_share or 0.0) > 0.5 or not a_holding
        i_b = next((i for i in range(i_open, len(records)) if _touching(records[i], arm_b)), None)
        if i_b is not None:
            handoff_gap = float(t[i_b] - t[i_open])
        # Transfer: from the latch opening until the panel first rests on the mat (or the end).
        i_place = next((i for i in range(i_open, len(records)) if foot_on_mat(records[i], layout)), len(records) - 1)
        window = records[i_open : i_place + 1]
        both = [_touching(r, arm_a) and _touching(r, arm_b) for r in window]
        two_hand_share = float(np.mean(both)) if both else 0.0

    tilts = [unplanned_tilt_deg(r["panel_quat"], rest_quat) for r in records]
    ang = np.array([r["panel_ang_vel"] for r in records])
    peak_off_axis_rate = float(np.degrees(np.max(np.linalg.norm(ang[:, [0, 2]], axis=1))))
    max_tilt = float(max(tilts))
    score, stage, failure = benchmark_score(records, layout, dropped=dropped, upright_on_mat=_upright_on_mat(records, layout))
    if score == 0.75 and task_success:  # both grippers on it as well
        score, stage = STAGES[3]
    return {
        "benchmark_score": score,
        "benchmark_stage": stage,
        "benchmark_failure": failure,
        "task_success": task_success,
        "monitor_dropped": dropped,
        "release_load_share": release_load_share,
        "premature_release": bool(premature),
        "handoff_gap_s": handoff_gap,
        "two_hand_share": two_hand_share,
        "max_unplanned_tilt_deg": max_tilt,
        "peak_off_axis_rate_dps": peak_off_axis_rate,
        "safe_success": bool(task_success and not dropped and not premature and max_tilt <= UNPLANNED_TOL_DEG),
        "max_drop_speed": float(np.max(down)),
        "final_screen_up_deg": screen_up_deg(final["panel_quat"]),
        "final_on_mat": bool(on_mat),
        "final_speed": float(np.linalg.norm(final["panel_lin_vel"])),
        "goal": goal,
    }
