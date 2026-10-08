"""Reproduce the manuscript's deterministic FSRS-6 scheduling-sensitivity example."""
from __future__ import annotations

import numpy as np
from srslib.memory import interval_multiplier

TARGET = 0.80
DEFAULT_PSI = 0.1542
DELTA_PSI = 0.05
PSI = np.array([DEFAULT_PSI - DELTA_PSI, DEFAULT_PSI, DEFAULT_PSI + DELTA_PSI])

if __name__ == "__main__":
    k = interval_multiplier(TARGET, PSI, family="power")
    for psi, val in zip(PSI, k):
        print(f"psi={psi:.4f}: interval={val:.6f} * S")
    lo, hi = float(k.min()), float(k.max())
    print(f"target retention = {TARGET:.2f}")
    print(f"range: {lo:.6f}S to {hi:.6f}S")
    print(f"longest-vs-shortest difference: {(hi/lo - 1.0)*100:.2f}%")
