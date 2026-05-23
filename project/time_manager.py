import numpy as np


class Clock:
    def __init__(self, total=300.0):
        self.initial_time = total
        self.time_left = total

    def spend(self, t):
        self.time_left = max(0.0, self.time_left - t)

    def ratio(self):
        return self.time_left / self.initial_time

    def flag(self):
        return self.time_left <= 0.0


class TimeManager:

    ARM_MULTIPLIERS = [
        0.5,
        1.0,
        2.0,
        4.0
    ]

    def compute_time_budget(
        self,
        arm,
        remaining_time,
        legal_moves,
        move_number,
        endgame
    ):

        expected_remaining_moves = 20 if endgame else 40

        base_time = remaining_time / expected_remaining_moves

        complexity = legal_moves / 30.0
        complexity = np.clip(complexity, 0.5, 2.0)

        arm_factor = self.ARM_MULTIPLIERS[arm]

        budget = base_time * complexity * arm_factor

        budget = min(budget, remaining_time * 0.25)

        budget = max(budget, 0.01)

        return budget
