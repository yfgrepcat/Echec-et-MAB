import numpy as np


# Clock class to manage time for each move
class Clock:
    """Class to manage the time left for the game,
    allowing to spend time on moves and check if the clock has ran out.

    :return: A Clock object to manage the time for the game.
    :rtype: Clock
    """

    # Each game is 5 minutes by default
    def __init__(self, total=300.0):
        self.initial_time = total
        self.time_left = total

    # Spend time t on the clock, ensuring it doesn't go negative
    def spend(self, t):
        self.time_left = max(0.0, self.time_left - t)

    # Get the ratio of time left to initial time, useful for scaling budgets
    def ratio(self):
        return self.time_left / self.initial_time

    # Check if the clock is ran out
    def flag(self):
        return self.time_left <= 0.0


# TimeManager computes the time budget for each move based on the arm, remaining time, legal moves, move number and endgame status
class TimeManager:
    """TimeManager computes a time-budget.
    This time-budget is used to limit the time spent on each move,
    ensuring that the model learns to make decisions within a reasonable time frame,
    which is crucial for real-time applications like chess.
    
    :return: A TimeManager object to compute time budgets for moves.
    :rtype: TimeManager
    """

    def __init__(self, arm0=0.05, arm1=0.3, arm2=1.0, arm3=3.0):
        # Fixed budgets in seconds, to provide a genuine time-allocation tradeoff.
        self.ARM_BUDGETS = [
            arm0,  # Arm 0: blitz / save clock (~0.05s)
            arm1,  # Arm 1: light (~0.3s)
            arm2,  # Arm 2: normal (~1.0s)
            arm3,  # Arm 3: deep (~3.0s, unaffordable every move)
        ]

    # Compute the time budget for a given move based on game state and arm choice
    def compute_time_budget(self, arm, remaining_time, legal_moves, endgame):
        fixed_budget = self.ARM_BUDGETS[arm]
        
        # We cap the fixed budget by a fraction of the remaining time to avoid immediate flagging,
        # but not too restrictive so that picking arm 3 when time is low will still flag.
        budget = min(fixed_budget, remaining_time * 0.5)
        
        # Minimum absolute floor to prevent 0.0 limit
        budget = max(budget, 0.01)
        return budget
