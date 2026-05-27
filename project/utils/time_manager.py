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


# TimeManager computes the time budget associated with each arm.
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

    def __init__(self, arm0=0.05, arm1=0.3, arm2=1.0, arm3=3.0, proportional_floor=0.001):
        # Fixed think-times make the arms easy to interpret and keep them separated.
        # On a 60s clock, always choosing arm 3 is intentionally unaffordable.
        self.ARM_BUDGETS = [
            arm0,  # Arm 0: blitz / save clock
            arm1,  # Arm 1: light search
            arm2,  # Arm 2: normal search
            arm3,  # Arm 3: deep search
        ]
        self.proportional_floor = proportional_floor

    # Compute the time budget for a given move based on game state and arm choice
    def compute_time_budget(self, arm, remaining_time, legal_moves, endgame):
        del legal_moves, endgame
        fixed_budget = self.ARM_BUDGETS[arm]
        floor = max(0.01, remaining_time * self.proportional_floor)
        return max(fixed_budget, floor)
