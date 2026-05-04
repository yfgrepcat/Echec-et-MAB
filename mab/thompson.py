"""
Not sure this will fit our needs but Thompson sampling is a popular algorithm for multi-armed bandits.
It could be a good starting point for our chess AI selection problem. 
It balances exploration and exploitation by sampling from the posterior distribution of each arm's success probability, 
which allows it to adaptively choose the best arm over time based on observed rewards (wins/losses).
"""
from typing import Any
from dataclasses import dataclass
import numpy as np

@dataclass
class ThompsonSampling:
    """Thompson sampling multi-armed bandit."""
    def __init__(self, n_arms: int, seed: int = 42):
        self.n_arms = n_arms
        self._rng = np.random.default_rng(seed)
        # Successes and failures track the number of times each arm (chess AI) has been successful (won) or failed (lost)
        # Alpha = successes + 1, Beta = failures + 1 for the Beta distribution
        self._alpha = np.zeros(n_arms, dtype=int)
        self._beta = np.zeros(n_arms, dtype=int)

    def select(self) -> int:
        # Sample from the Beta distribution for each arm
        samples = self._rng.beta(self._alpha, self._beta)
        # Choose the arm with the highest sample
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float) -> None:
        r = float(np.clip(reward, 0.0, 1.0))
        self._alpha[arm] += r
        self._beta[arm] += 1.0 - r


    def stats(self) -> dict[str, Any]:
        # The mean of the Beta distribution for each arm is alpha / (alpha + beta), which represents our current estimate of the probability of success for that arm (chess AI)
        means = self._alpha / (self._alpha + self._beta)
        return {
            "alpha": self._alpha.copy(),
            "beta": self._beta.copy(),
            "means": means,
        }
