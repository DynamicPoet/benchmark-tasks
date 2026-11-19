"""Call tick() exactly once per control step, from success(), which the harness calls every step. It keeps two records, both written only if TASK_STEP_FILE is set:

- $TASK_STEP_FILE: the simulated time. tools/run_dream.sh uses it to tell a harmless session expiry in the warm-up from a lost run.
- $TASK_STEP_FILE.worst30: the slowest 30 control steps so far, in seconds of wall time. Use it to choose Dream-1's rows per request.
"""

from __future__ import annotations

import os
import time

CONTROL_HZ = 20.0
_state = {"step": 0, "mark": None, "worst": 0.0}


def _write(path: str, text: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def tick() -> None:
    path = os.environ.get("TASK_STEP_FILE")
    if not path:
        return
    _state["step"] += 1
    step = _state["step"]
    _write(path, f"{step / CONTROL_HZ:.2f}")
    if step % 30 == 0:
        now = time.monotonic()
        if _state["mark"] is not None:
            _state["worst"] = max(_state["worst"], now - _state["mark"])
            _write(f"{path}.worst30", f"{_state['worst']:.2f}")
        _state["mark"] = now
