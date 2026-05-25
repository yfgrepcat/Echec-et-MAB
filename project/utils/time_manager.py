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
    The time budget is computed based on the current game state,
    including the remaining time, the number of legal moves, the move number,
    and whether the game is in the endgame phase.

    :return: A TimeManager object to compute time budgets for moves.
    :rtype: TimeManager
    """

    def __init__(self, arm0=0.5, arm1=1.0, arm2=2.0, arm3=4.0):
        self.ARM_MULTIPLIERS = [
            arm0,  # Arm 0 gets the base time budget
            arm1,  # Arm 1 gets twice the base time budget
            arm2,  # Arm 2 gets four times the base time budget
            arm3,  # Arm 3 gets eight times the base time budget
        ]

    # Compute the time budget for a given move based on game state and arm choice
    def compute_time_budget(self, arm, remaining_time, legal_moves, endgame):
        expected_remaining_moves = 20 if endgame else 40  # Standard chess heuristic
        base_time = (
            remaining_time / expected_remaining_moves
        )  # In average, we want to spend and equal fraction of remaining time
        # --> if startgame, we have time to explore, in endgame not
        complexity = (
            legal_moves / 30
        )  # 30 legal moves is a very complex position, 10 legal moves is a simple position
        # --> for a lot of legal move, the complexity is high so more time is needed to choose arm wisly
        complexity = np.clip(
            complexity, 0.5, 2.0
        )  # Limit complexity factor to avoid taking too much time
        arm_factor = self.ARM_MULTIPLIERS[
            arm
        ]  # Arm normalization factor, higher arms get more time
        budget = base_time * complexity * arm_factor  # Budget is calculated
        budget = min(
            budget, remaining_time * 0.25
        )  # Don't allow spending more than 25% of remaining time on a single move
        budget = max(budget, 0.01)  # To avoid flaging immediately (minimum 100ms)
        return budget
