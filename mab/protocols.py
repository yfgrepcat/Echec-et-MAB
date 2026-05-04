from typing import Protocol, runtime_checkable, Any
import numpy as np

@runtime_checkable
class Bandit(Protocol):
    """
    A multi-armed bandit selects arms and learns from rewards.

    Two main methods:
     1. select() - pick an arm index
     2. update() - observe the reward, adjuts internal beliefs
    """

    def select(self) -> int: ...

    def update(self, arm: int, reward: float) -> None: ...

    def stats(self) -> dict[str, Any]: ...


@runtime_checkable
class ContextualBandit(Protocol):
    """
    A contextual multi-armed bandit: selects arms and learns from (context, reward) pairs.
    Context is a 1-D numpy feature vector of shape (d,) encoding current game state.
    """

    def select(self, context: np.ndarray) -> int: ...

    def update(self, arm: int, reward: float, context: np.ndarray) -> None: ...

    def stats(self) -> dict[str, Any]: ...