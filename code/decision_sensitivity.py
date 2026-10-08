"""Reproduce the manuscript's deterministic FSRS-6 scheduling sensitivity example."""
from __future__ import annotations
import numpy as np
from srslib.memory import interval_multiplier

TARGET = 0.80
PSI = np.array([0.10, 0.80], dtype=float)

if __name__ == "__main__":
    k = interval_multiplier(TARGET, PSI, family="power")
    lo, hi = float(k.min()), float(k.max())
    print(f"target retention = {TARGET:.2f}")
    for psi, val in zip(PSI, k):
        print(f"psi={psi:.2f}: interval={val:.6f} * S")
    print(f"range: {lo:.6f}S to {hi:.6f}S")
    print(f"relative spread: {(hi/lo - 1.0)*100:.2f}%")
