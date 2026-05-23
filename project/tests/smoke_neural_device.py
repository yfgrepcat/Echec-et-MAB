#!/usr/bin/env python3
"""Smoke test for device handling in NeuralLinUCB via ChessMAB.

Runs two configurations:
 - force CPU
 - auto (let PyTorch pick)

Usage: python3 project/tests/smoke_neural_device.py
"""

import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mab_agent import ChessMAB


def try_config(cfg):
    print("Testing config:", cfg)
    mab = ChessMAB(engine=None, model_path="models/smoke_neural_dev.pt", bandit_type="neural_linucb", bandit_config=cfg)
    x = np.zeros((7, 1))
    arm = mab.bandit.select_arm(x)
    print(" selected arm", arm)
    mab.bandit.update(arm, x, 0.5)
    print(" update ok\n")


def main():
    configs = [
        {"device": "cpu", "force_cpu": True},
        {"device": "auto", "force_cpu": False},
    ]

    for c in configs:
        try_config(c)

    print("All device smoke tests passed")


if __name__ == "__main__":
    main()
