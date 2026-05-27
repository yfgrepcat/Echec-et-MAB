from __future__ import annotations

import chess
import chess.engine
import numpy as np
import os
import time

from neural_linucb import NeuralLinUCB
from basic_linucb import LinUCB
from utils.utils import material_balance, is_endgame
from utils.time_manager import TimeManager

#TODO: Use Pydantic for validation.
def sanitize_bandit_config(bandit_config: dict | None) -> dict:
    """ Method used to sanitize the bandit configuration, ensuring that the provided configuration is valid and contains recognized keys with appropriate values. This is especially important for the neural LinUCB implementation, which may have specific requirements for hardware usage (e.g., GPU vs CPU). The method checks for the presence of expected keys, validates their values, and provides defaults when necessary. It raises errors if unrecognized keys are found or if values are of incorrect types, helping to prevent silent failures or misconfigurations that could lead to suboptimal performance or crashes during training.

    :param bandit_config: Configuration dictionary for the bandit algorithm, which may include keys like 'device' 
        to specify hardware usage (e.g., 'auto', 'cpu', 'cuda', 'mps') and 'force_cpu' 
        to indicate whether to force CPU usage even if a GPU is available. 
        This allows users to customize the behavior of the neural LinUCB implementation 
        based on their hardware capabilities and preferences.
    :type bandit_config: dict | None
    :raises ValueError: If invalid values for 'device' or 'force_cpu'.
    :return: Sanitized configuration dictionary with validated and defaulted values for the bandit algorithm.
    :rtype: dict
    """
    cfg = {} if bandit_config is None else dict(bandit_config) # Copy of the config
    device = cfg.get("device", "auto") # Default to 'auto' to let the library choose the best hardware available (GPU if possible, else CPU)
    if device is None: device = "auto" # Handle None value for device, treat it as 'auto'
    device = device.lower() # Normalize device string to lowercase for consistency          
    if device not in ("auto", "cpu", "cuda", "mps"): # Raise an error if the key is not recognized
        raise ValueError("bandit_config.device must be one of 'auto','cpu','cuda','mps'")
    force_cpu = cfg.get("force_cpu", False) # Handling weak PCs
    if not isinstance(force_cpu, bool):                                                         
        raise ValueError("bandit_config.force_cpu must be a boolean")
    return {"device": device, "force_cpu": force_cpu} # Return sanitized config 

class ChessMAB:
    def __init__(self, engine, model_path="models/final_model.npz", bandit_config: dict | None = None, bandit_type="basic_linucb"):
        """ Initialize the ChessMAB agent with the chess engine, model path, bandit type and configuration. 
        The constructor sets up the chess engine for move generation and position evaluation, 
        initializes the bandit algorithm based on the specified type (basic LinUCB or neural LinUCB), 
        and prepares the time manager for computing time budgets. 
        It also attempts to load existing model parameters from the specified file path, 
        allowing for continued training or evaluation from a previously saved state.

        :param engine: Chess engine instance used for move generation and position evaluation.
        :type engine: chess.engine.SimpleEngine or compatible chess engine instance
        :param model_path: Path to the model file for saving/loading parameters, defaults to "models/final_model.npz"
        :type model_path: str, optional
        :param bandit_config: Configuration dictionary for the bandit algorithm, defaults to None
        :type bandit_config: dict | None, optional
        :param bandit_type: Type of the bandit algorithm to use, defaults to "basic_linucb"
        :type bandit_type: str, optional
        """

        self.engine = engine
        self.model_path = model_path
        self.bandit_type = bandit_type
        self.bandit_config = bandit_config or {}
        self.n_features = 7 # 7 features for the context (legal moves, time left, move number, material balance, is endgame, is in check, captures)
        self.n_arms = 4 # 4 arms corresponding to 4 time budget categories (very short (0), short (1), medium (2), long (3)) 
        if self.bandit_type == "basic_linucb":
            self.bandit = LinUCB(self.n_arms, self.n_features)
        elif self.bandit_type == "neural_linucb": # With neural network to learn a better representation of the context features
            self.bandit = NeuralLinUCB(
                n_arms=self.n_arms,
                n_features=self.n_features,
                **self.bandit_config,
            )
        else:
            raise ValueError(f"Unsupported bandit_type: {self.bandit_type}")
        self.time_manager = TimeManager() # Time manager to compute time budgets for each move based on the selected arm and the remaining time, legal moves, etc. 
        self.load()

    def save(self):
        """
        Save the model parameters to a file. For the basic LinUCB implementation, it saves the A_inv and b parameters for each arm in a .npz file. 
        For the neural LinUCB implementation, it delegates the saving process to the neural bandit's own save method, 
        The method ensures that the model is saved safely by writing to a temporary file first and then replacing the original file, reducing the risk of corruption during the save process.
        """
        if self.bandit_type == "neural_linucb":
            # delegate save to neural implementation
            self.bandit.save(self.model_path)
            return
        temp_path = self.model_path + ".tmp.npz" # Save to as temporary file first to avoid corruption of the model original file
        np.savez(temp_path, A_inv=self.bandit.A_inv, b=self.bandit.b) # Save the model parameters (A_inv and b for each arm)
        os.replace(temp_path, self.model_path) # Replace original model file by the new one

    def load(self):
        """
        Load the model parameters from a file if it exists. 
        For the basic LinUCB implementation, it loads the A_inv and b parameters for each arm from a .npz file.
        """
        if not os.path.exists(self.model_path):
            return
        try:
            if self.bandit_type == "neural_linucb":
                try:
                    self.bandit.load(self.model_path)
                    print(f"Loaded neural model: {self.model_path}")
                except Exception:
                    print("Failed to load neural model; continuing with fresh model.")
                return
            data = np.load(self.model_path) # Load the model parameters from the .npz file
            self.bandit.A_inv = list(data['A_inv']) # Set A_inv for each arm from the loaded data
            self.bandit.b = list(data['b']) # Set b for each arm from the loaded data                 
            print(f"Loaded model: {self.model_path}")   
        except Exception:
            print("Corrupted model file.")

    def evaluate(self, board, depth=8):
        """ Method to evaluate a chess position using the chess engine.
        It analyzes the given board position to obtain a score, which is then 
        converted into a WDL (Win/Draw/Loss) probability for the current player.

        :param board: Chess board position to evaluate.
        :type board: chess.Board
        :param depth: Search depth for the chess engine, defaults to 8
        :type depth: int, optional
        :return: WDL score representing the probability of winning for the current player (1.0 = winning position, 0.0 = losing position, 0.5 = balanced position)
        :rtype: float
        """
        info = self.engine.analyse(
            board,
            chess.engine.Limit(depth=depth)
        )
        # Get the score from analysis. (positive = white advantage, negative = black advantage)
        score = info["score"].white()
        
        # Inverse the score if it's black
        if board.turn == chess.BLACK:
            score = -score
            
        if score.is_mate():
            mate_moves = score.mate()
            if board.turn == chess.BLACK:
                mate_moves = mate_moves
            eval_cp = 10000 if mate_moves > 0 else -10000
        else:
            eval_cp = score.score() # If it's not a mate score, we take the centipawn evaluation directly (positive if white is better, negative if black is better)
        wdl = 1.0 / (1.0 + 10.0 ** (-eval_cp / 400.0)) # Convert centipawn score to WDL (Win/Draw/Loss) probability using a logistic function.
            #   - wdl = 1.0: player has a winning position
            #   - wdl = 0.0: player has a losing position
            #   - wdl = 0.5: Balanced position 
            # TODO : Explain each part of 1.0 / (1.0 + 10.0 ** (-eval_cp / 400.0)) in rapport
        return wdl

    def extract_features(self, board, clock):
        """ Method to extract features from the chess board and clock for the context representation used by the bandit algorithm.
        The method computes several features that capture important aspects of the current game state, including:
            - Number of legal moves available for the current position
            - Remaining time as a ratio of the initial time
            - Move number in the game
            - Material balance between the two players
            - Whether the position is in the endgame phase
            - Whether the current player is in check
            - Number of capture moves available

        :param board: _description_
        :type board: _type_
        :param clock: _description_
        :type clock: _type_
        :return: _description_
        :rtype: _type_
        """
        legal_moves = len(list(board.legal_moves))
        captures = len([m for m in board.legal_moves if board.is_capture(m)]) # Number of legal moves that are captures (eats an opponent piece)
        # Our context
        features = np.array([
            legal_moves / 50.0,                  # Current position, normalized by a constant (e.g., 50) to keep it in a reasonable range for the model
            clock.ratio(),                       # Remaining time as a ratio of the initial time (between 0 and 1)
            board.fullmove_number / 100.0,       # Move number in game, normalized by 100 to keep every parameter in same range
            material_balance(board) / 39.0,      # Difference in piece values between the two players, normalized by the maximum possible imbalance : 39, to keep it between -1 and 1
            int(is_endgame(board)),              # 1 if is endgame, 0 else
            int(board.is_check()),               # 1 if the current player is in check, 0 else                  
            captures / 10.0                      # Normalized number of captures
        ])      
        return features.reshape(-1, 1) # Reshape into a column vector: Array -> Column vector (n_features x 1)

    def compute_reward(self, board: chess.Board, move: chess.Move, elapsed: float) -> float:
        """ Method to compute the reward for a move based on the change in position evaluation before and after the move,
        and the time taken to play the move. The reward is calculated by evaluating the position before
        and after playing the move, computing the change in WDL score (which reflects the improvement or deterioration of the position),
        and then normalizing this change to give more weight to significant improvements.
        Additionally, the method penalizes long move times to encourage the agent to play efficiently.

        :param board: Chess board position before playing the move, used for evaluation to compute the reward based on the change in position quality.
        :type board: chess.Board
        :param move: The move played by the engine, represented as a chess.Move object, used to update the board position for reward computation.
        :type move: chess.Move
        :param elapsed: Time taken to play the move, used to penalize long move times in the reward calculation.
        :type elapsed: float
        :return: Computed reward for the move, which reflects the improvement in position quality and penalizes long move times.
        :rtype: float
        """
        score_before = self.evaluate(board, depth=8) # Evaluate the position before playing the move (Perspective: Joueur A)
        board.push(move)
        score_after = self.evaluate(board, depth=8) # Evaluate the position after playing the move (Perspective: Joueur B)
        board.pop() # Undo the move to restore the original position (for other move evaluations)
        # delta_wdl is the change in White's WDL probability; naturally bounded in [-1, +1].
        # Quality and terminal both live on the same [-1, +1] scale so neither dominates by magnitude alone.
        delta_wdl = (1.0 - score_after) - score_before # On ajuste car score_after est du point de vue de B. 1 - score_after redonne la probabilité pour A.
        # Time penalty capped at 0.1 -- earlier 0.5 cap was ~5x larger than the typical quality
        # difference between arms, which made the bandit prefer "always play fast" even when
        # longer thinking produced measurably better moves.
        time_penalty = 0.015 * min(elapsed / 1.5, 1.0)
        reward = delta_wdl - time_penalty
        return reward

    def compute_terminal_reward(self, board: chess.Board, clock_flagged: bool = False) -> float:
        """ Terminal reward from White's perspective, called once per game when the game has ended.
        clock_flagged=True overrides the result and applies the loss-on-time penalty,
        since python-chess does not know about our custom Clock.
        Rescaled to +-1.0 to live on the same scale as per-move delta_wdl.

        :param board: Final chess board state at game end.
        :type board: chess.Board
        :param clock_flagged: True if the MAB clock ran out, defaults to False
        :type clock_flagged: bool, optional
        :return: +1.0 White win, -1.0 White loss (or flag), 0.0 draw / not yet over.
        :rtype: float
        """
        if clock_flagged:
            return -1.0
        if not board.is_game_over(claim_draw=True):
            return 0.0
        result = board.result(claim_draw=True)
        if result == "1-0":
            return 1.0
        if result == "0-1":
            return -1.0
        return 0.0

    def play(self, board, clock, training=True) -> tuple[chess.Move, int, float, float, float, np.ndarray]:
        """ Method to select and play a move based on the current board position and clock state.
        In training mode, the bandit update is deferred to the caller (training loop) so the
        terminal reward can be credited to the last White ply regardless of whether the game
        ends on White's or Black's move (otherwise losses-by-Black-mate go uncredited).

        :param board: Current chess board state.
        :type board: chess.Board
        :param clock: Clock object representing the remaining time for the current player.
        :type clock: Clock
        :param training: Flag indicating whether the agent is in training mode, defaults to True
        :type training: bool, optional
        :return: A tuple (move, arm, per-move reward, elapsed, budget, context vector x).
            In training mode, the caller is responsible for adding terminal/flag rewards and calling bandit.update.
        :rtype: tuple[chess.Move, int, float, float, float, np.ndarray]
        """

        x = self.extract_features(
            board,
            clock
        )
        arm = self.bandit.select_arm(x)
        legal_moves = len(list(board.legal_moves))
        budget = self.time_manager.compute_time_budget(
            arm=arm,
            remaining_time=clock.time_left,
            legal_moves=legal_moves,
            endgame=is_endgame(board)
        )
        start = time.time() # Start time to measure elapsed time for playing the move
        # Play the move using the chess engine WITHIN the computed time budget
        result = self.engine.play(
            board,
            chess.engine.Limit(time=budget)
        )
        elapsed = time.time() - start
        move = result.move
        reward = 0.0
        if training:
            reward = self.compute_reward(
                board,
                move,
                elapsed
            )
        clock.spend(elapsed) # Update the clock by spending the elapsed time for playing the move
        return (move, arm, reward, elapsed, budget, x) # x is returned so the training loop can flush a deferred update with terminal reward
