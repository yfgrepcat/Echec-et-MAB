#!/usr/bin/env python3
"""Smoke test: instantiate ChessMAB with neural_linucb and run select/update cycle.

Run with: python3 project/tests/smoke_neural.py

Requires: torch installed in the environment.
"""

import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mab_agent import ChessMAB


def run():
    print("Creating ChessMAB with neural_linucb (smoke test)")
    mab = ChessMAB(engine=None, model_path="models/smoke_neural.pt", bandit_type="neural_linucb")

    # simple zero-context
    x = np.zeros((7, 1))

    arm = mab.bandit.select_arm(x)
    print("Selected arm:", arm)

    mab.bandit.update(arm, x, 1.0)
    print("Update applied successfully")

    print("Smoke test passed")


if __name__ == "__main__":
    run()
