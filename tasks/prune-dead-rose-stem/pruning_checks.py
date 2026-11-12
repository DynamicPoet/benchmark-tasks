"""Cut history and final-state checks for retaining a living bud."""

from dataclasses import dataclass
import math


@dataclass
class PruningState:
    bud_top: float
    first_cut: float | None = None
    stub_length: float | None = None
    bud_damaged: bool = False
    cuts: int = 0

    def cut(self, height: float) -> None:
        """Record every cut through the remaining cane; a cut cannot grow the stub."""
        if not math.isfinite(height):
            self.bud_damaged = True
            return
        if self.first_cut is None:
            self.first_cut = height
        self.stub_length = height if self.stub_length is None else min(self.stub_length, height)
        self.bud_damaged |= height <= self.bud_top
        self.cuts += 1

    def observe_bud(self, attached: bool) -> None:
        """A bud lost at any time stays lost, even if an object is put back."""
        self.bud_damaged |= not attached

    def credit(self, final_stub: float, bud_attached: bool, good_range=(0.003, 0.015)) -> float:
        """Plant credit: no cut or damaged bud 0; misplaced first/final cut .25; both good 1."""
        if (self.first_cut is None or self.bud_damaged or not bud_attached
                or not math.isfinite(final_stub) or final_stub <= self.bud_top):
            return 0.0
        lo, hi = good_range
        first_good = lo <= self.first_cut - self.bud_top <= hi
        final_good = lo <= final_stub - self.bud_top <= hi
        return 1.0 if first_good and final_good else 0.25
