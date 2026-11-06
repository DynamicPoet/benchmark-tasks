"""On the station table, in front of the arms and centred between them: a 2 kg dummy monitor (48 cm wide, 20 cm
tall, 3 cm thick, black with a blue screen on the side facing away from the robot) hangs on a quick-release
bracket on the front of a graphite stand's neck, its bottom edge 8 cm above the table. An orange release lever
sticks out of the robot-facing side of the neck, right of centre; pushing it forward 3 mm releases the latch.
A green foam mat (33 x 50 x 2 cm) lies just in front of the monitor.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:    python demo.py task.py demo_out

The objects, the latch, the scoring and the telemetry live in robocurve.arena.monitor_detach (src/ next to this file).
"""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
if (HERE / "src").is_dir():
    sys.path.insert(0, str(HERE / "src"))

from robocurve.arena.monitor_detach import OBJECTS, SCENE, SCORE_OVER, SUCCESS_HOLD_S, score, success  # noqa: E402

INSTRUCTION = ("Take the monitor off its stand and set it down upright on the mat, keeping hold of it. "
               "The release lever is on the stand's neck.")

__all__ = ["INSTRUCTION", "OBJECTS", "SCENE", "SCORE_OVER", "SUCCESS_HOLD_S", "score", "success"]
