import numpy as np

# Clock class to manage time for each move
class Clock:

    # Each game is 5 minutes by default
    def __init__(self, total=300.0):
        self.initial_time = total 
        self.time_left = total

    # Spend time t on the clock, ensuring it doesn't go negative
    def spend(self, t): self.time_left = max(0.0, self.time_left - t)

    # Get the ratio of time left to initial time, useful for scaling budgets
    def ratio(self): return self.time_left / self.initial_time

    # Check if the clock is ran out
    def flag(self): return self.time_left <= 0.0

# TimeManager computes the time budget for each move based on the arm, remaining time, legal moves, move number and endgame status
class TimeManager:

    # Higher arms should get more time, so we need to add multipliers to adapt the high time budget 
    # Each arm ger twice the time requiered for the previous one
    ARM_MULTIPLIERS = [
        0.5,    # Arm 0
        1.0,    # Arm 1
        2.0,    # Arm 2
        4.0     # Arm 3
    ]

    # Compute the time budget for a given move based on game state and arm choice
    def compute_time_budget(self, arm, remaining_time, legal_moves, move_number, endgame):
        expected_remaining_moves = 20 if endgame else 40            # Values are based on guess so mabe need adjustment (TODO)
        base_time = remaining_time / expected_remaining_moves       # In average, we want to spend and equal fraction of remaining time
                                                                    # --> if startgame, we have time to explore, in endgame not
        complexity = legal_moves / 30.0                             # 30 legal moves is a very complex position, 10 legal moves is a simple position
                                                                    # --> for a lot of legal move, the complexity is high so more time is needed to choose arm wisly
        complexity = np.clip(complexity, 0.5, 2.0)                  # Limit complexity factor to avoid taking too much time
        arm_factor = self.ARM_MULTIPLIERS[arm]                      # Arm normalization factor, higher arms get more time
        budget = base_time * complexity * arm_factor                # Budget is calculated 
        budget = min(budget, remaining_time * 0.25)                 # Don't allow spending more than 25% of remaining time on a single move    
        budget = max(budget, 0.01)                                  # To avoid flaging immediately (minimum 100ms)
        return budget
