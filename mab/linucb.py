from __future__ import annotations

from typing import Any
from dataclasses import dataclass

import numpy as np


@dataclass
class LinUCB:
    """LinUCB disjoint contextual bandit.

    :param n_arms: Number of arms (chess algorithms to choose from).
    :param d: Dimensionality of the context feature vector.
    :param alpha: Exploration-exploitation trade-off. Higher values favour exploration. Defaults to 1.0.
    :param seed: Random seed for tie-breaking. Defaults to 42.
    """

    def __init__(self, n_arms: int, d: int, alpha: float = 1.0, seed: int = 42) -> None:
        self.n_arms = n_arms
        self.d = d
        self.alpha = alpha
        self._rng = np.random.default_rng(seed)
        # _covariance[a] tracks which feature directions we've seen.
        # _reward_signal[a] tracks reward-weighted features.
        self._covariance = [np.eye(d) for _ in range(n_arms)]
        self._reward_signal = [np.zeros(d) for _ in range(n_arms)]
        self._last_context: np.ndarray | None = None
        self._counts: np.ndarray = np.zeros(n_arms, dtype=int)

    def select(self, context):
        x = np.asarray(context, dtype=float)
        self._last_context = x.copy()
        
        scores = np.empty(self.n_arms)
        for a in range(self.n_arms):
            # Solve: theta = covariance⁻¹ @ reward_signal  (the learned weights)
            theta = np.linalg.solve(self._covariance[a], self._reward_signal[a])
            
            # Predicted reward = dot product of weights and context
            predicted_reward = theta @ x
            
            # Uncertainty: how unfamiliar is this context for this arm?
            # Large when we haven't seen contexts like x for arm a yet.
            cov_inv_x = np.linalg.solve(self._covariance[a], x)
            uncertainty = np.sqrt(x @ cov_inv_x)
            
            # UCB: be optimistic — score = mean estimate + exploration bonus
            scores[a] = predicted_reward + self.alpha * uncertainty
        
        # Break ties randomly (avoid bias toward arm 0 early on)
        best_score = scores.max()
        candidates = np.where(np.isclose(scores, best_score))[0]
        return int(self._rng.choice(candidates))

    def update(self, arm, reward, context):
        x = np.asarray(context, dtype=float)
        
        # Accumulate: "I saw this context shape" (rank-1 update)
        self._covariance[arm] += np.outer(x, x)
        
        # Accumulate: "I got this reward in this context"
        self._reward_signal[arm] += reward * x
        
        self._counts[arm] += 1

    def stats(self) -> dict[str, Any]:
        thetas = [np.linalg.solve(self._covariance[a], self._reward_signal[a]) for a in range(self.n_arms)]

        ucb_scores = None
        if self._last_context is not None:
            x = self._last_context
            ucb_scores = np.array(
                [
                    thetas[a] @ x
                    + self.alpha * np.sqrt(x @ np.linalg.solve(self._covariance[a], x))
                    for a in range(self.n_arms)
                ]
            )

        return {
            "counts": self._counts.copy(),
            "theta": [t.copy() for t in thetas],
            "ucb_scores": ucb_scores,
            "alpha": self.alpha,
        }
