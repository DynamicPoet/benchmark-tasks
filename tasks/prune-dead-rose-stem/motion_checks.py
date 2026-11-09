"""Motion checks for points sampled at a fixed control rate."""

from collections.abc import Sequence

import numpy as np


def points_at_rest(history: Sequence[np.ndarray], max_move: float, steps: int) -> bool:
    """Every point must move less than max_move between any two samples in the window."""
    if len(history) <= steps:
        return False
    points = np.asarray(history[-(steps + 1):], dtype=np.float64).reshape(steps + 1, -1, 3)
    displacement = points[:, None] - points[None, :]
    return bool(np.all(np.linalg.norm(displacement, axis=-1) < max_move))
