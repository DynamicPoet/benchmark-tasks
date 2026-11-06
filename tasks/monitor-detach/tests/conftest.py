"""Run the tests from a checkout without installing it: the code is in src/ next to tests/."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if SRC.is_dir():
    sys.path.insert(0, str(SRC))
