from typing import Any
from dataclasses import dataclass
import numpy as np


@dataclass
class EpsilonGreedy:
    """Epsilon-greedy multi-armed bandit.

    :param epsilon: The probability of exploring (choosing a random arm) instead of exploiting (choosing the best-known arm).
    :type epsilon: float, optional
    """
    def __init__(self, n_arms: int, epsilon: float = 0.15, decay: float = 0.999, seed: int = 42):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.decay = decay
        self._rng = np.random.default_rng(seed)
        self._counts = np.zeros(n_arms, dtype=int)
        self._values = np.zeros(n_arms, dtype=float)

    def select(self) -> int:
        if self._rng.random() < self.epsilon:
            # Explore: choose a random arm
            return self._rng.integers(self.n_arms)
        else:
            # Exploit: choose the arm with the highest estimated value
            return int(np.argmax(self._values))

    def update(self, arm: int, reward: float) -> None:
        # Update the counts and values for the selected arm
        # Counts counts how many times we've pulled this arm (chose said chess AI)
        self._counts[arm] += 1
        n = self._counts[arm]
        # Value is the reward estimate for this arm (chess AI) based on the rewards we've observed so far
        value = self._values[arm]
        new_value = value + (reward - value) / n
        self._values[arm] = new_value
        # Decay epsilon to reduce exploration over time
        self.epsilon *= self.decay

    def stats(self) -> dict[str, Any]:
        return {
            "epsilon": self.epsilon,
            "counts": self._counts.copy(),
            "values": self._values.copy(),
        }