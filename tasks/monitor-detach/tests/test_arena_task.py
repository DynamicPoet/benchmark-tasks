"""The yam_arena task file loads without the simulator and scores its telemetry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

task_file = pytest.importorskip("yam_arena.task_file")

from test_fastener_scoring import _record  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
# The repository keeps the task file in submission/monitor-detach/; the benchmark's task folder has it at the top.
TASK = next(p for p in (_ROOT / "task.py", _ROOT / "submission" / "monitor-detach" / "task.py") if p.is_file())


def test_task_file_declares_the_station_objects() -> None:
    task = task_file.load(TASK)
    assert task.scene == "station" and task.names == ("monitor", "stand", "mat")
    assert task.free_objects == ("monitor",) and task.success is not None and task.score is not None


def test_instruction_matches_task_json() -> None:
    import json

    assert task_file.load(TASK).module.INSTRUCTION == json.loads((TASK.parent / "task.json").read_text())["instruction"]


def test_task_scores_its_telemetry() -> None:
    from robocurve.arena import monitor_detach as md
    from robocurve.sim.layout import LAYOUT

    h = LAYOUT.panel_size[2] / 2
    on_mat = (0.45, -0.305, LAYOUT.mat_top_z + h)
    md._STATE["records"] = [_record(0.0, (0.331, -0.305, 0.18)), _record(4.4, on_mat, latch=True), _record(5.0, on_mat, latch=True)]
    assert md.success({}) and md.score({}) == 1.0
    md._STATE["records"] = [_record(0.0, (0.331, -0.305, 0.18)), _record(0.5, (0.331, -0.305, 0.10), vz=-1.2, touching=False, latch=True)]
    assert not md.success({}) and md.score({}) == 0.0
    assert np.allclose(np.array(md.MONITOR.pos) - md.STATION, [0.331, -0.305, 0.18])
