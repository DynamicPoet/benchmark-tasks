"""Re-score trials offline from their telemetry JSONL files (no simulator needed).

    python rescore.py logs/arena_telemetry/*.jsonl [--json]

Prints each trial's benchmark score and stage, plus the diagnostic metrics behind it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for root in (HERE, HERE.parent):  # the task folder has src/ beside this file; the repository, one level up
    if (root / "src").is_dir():
        sys.path.insert(0, str(root / "src"))

from robocurve.scoring import score_trial  # noqa: E402
from robocurve.sim.layout import LAYOUT  # noqa: E402


def rescore(path: Path) -> dict:
    lines = path.read_text().splitlines()
    header, records = json.loads(lines[0]), [json.loads(line) for line in lines[1:]]
    arm_a = header.get("arm_a", "left")
    metrics = score_trial(records, LAYOUT, arm_a, "right" if arm_a == "left" else "left")
    return {"file": path.name, "scene": header.get("scene"), "steps": len(records), "t_end": records[-1]["t"], **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--json", action="store_true", help="Print every metric as JSON lines.")
    args = parser.parse_args()
    for path in args.files:
        m = rescore(path)
        if args.json:
            print(json.dumps(m))
            continue
        why = m["benchmark_failure"] or m["benchmark_stage"]
        print(f"{m['file']}: score {m['benchmark_score']:.2f} ({why}); {m['steps']} steps, {m['t_end']:.1f} s; "
              f"dropped={m['monitor_dropped']} premature_release={m['premature_release']} "
              f"load_at_release={m['release_load_share']}")


if __name__ == "__main__":
    main()
