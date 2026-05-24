import chess
import chess.engine
import numpy as np
import os
import time

from neural_linucb import NeuralLinUCB
from basic_linucb import LinUCB
from utils.utils import material_balance, is_endgame
from utils.time_manager import TimeManager

# Method used to sanitze the configuration so we can ensure that the config is valid, especially useful for the neural LinUCB
# Without sanitization, our first tests showed that the agent would break easily if errors in configuration are hidden
# Format of bandit config : {TODO}
# Recognized keys:
#   - device: one of 'auto' (default), 'cpu' (to force CPU usage - No GPU), 'cuda' (to force CUDA usage - NVIDIA GPUs), 'mps' (to force Metal Performance Shaders usage - Apple Silicon)
#   - force_cpu: bool (needed to make the agent work on N7's portable PC without GPU)
def sanitize_bandit_config(bandit_config: dict | None) -> dict:
    cfg = {} if bandit_config is None else dict(bandit_config)                                  # Copy of the config
    device = cfg.get("device", "auto")                                                          # Default to 'auto' to let the library choose the best hardware available (GPU if possible, else CPU)
    if device is None: device = "auto"                                                          # Handle None value for device, treat it as 'auto'
    device = device.lower()                                                                     # Normalize device string to lowercase for consistency          
    if device not in ("auto", "cpu", "cuda", "mps"):                                            # Raise an error if the key is not recognized
        raise ValueError("bandit_config.device must be one of 'auto','cpu','cuda','mps'")
    force_cpu = cfg.get("force_cpu", False)                                                     # Handling weak PCs
    if not isinstance(force_cpu, bool):                                                         
        raise ValueError("bandit_config.force_cpu must be a boolean")
    return {"device": device, "force_cpu": force_cpu}                                           # Return sanitized config 

# ChessMAB is our implementation of the multi-armed bandit agent
# It will use the LinUCB algorithm (above) with or without a neural network to select moves during an awsome game of chess
class ChessMAB:

    # Initialization of the ChessMAB agent with the chess engine, model path, bandit type and configuration
    def __init__(self, engine, model_path="models/final_model.npz", bandit_config: dict | None = None, bandit_type="basic_linucb"):

        self.engine = engine                        # Chess engine used to play moves and evaluate positions. We will use Stockfish for this, but other engines are possible to users
        self.model_path = model_path                # Path to save/load the model parameters (A_inv and b for each arm). .npz format to store multiple arrays (the model parameters) in a single file
        self.bandit_type = bandit_type              # WITH/WITHOUT neural network
        self.bandit_config = bandit_config or {}    # Configuration for the bandit algorithm
        self.n_features = 7                         # 7 features for the context (legal moves, time left, move number, material balance, is endgame, is in check, captures)
        self.n_arms = 4                             # 4 arms corresponding to 4 time budget categories (very short (0), short (1), medium (2), long (3)) 
        match self.bandit_type:                     
            case "basic_linucb":                                        # Basic one without neural network
                self.bandit = LinUCB(self.n_arms, self.n_features)      
            case "neural_linucb":                                       # With neural network to learn a better representation of the context features                                   
                self.bandit = NeuralLinUCB(
                    n_arms=self.n_arms,
                    n_features=self.n_features,
                    **self.bandit_config,
                )
            case _:                                                     # Default so guess what, basic LinUCB again
                self.bandit = LinUCB(self.n_arms, self.n_features)
        self.time_manager = TimeManager()                               # Time manager to compute time budgets for each move based on the selected arm and the remaining time, legal moves, etc. 
                                                                        # It will use a heuristic approach to compute the time budget for each arm (very short, short, medium, long) based on the game phase, number of legal moves, and remaining time
        self.load()                                                     # Load model parameters from file if it exists, otherwise initialize a new model

    # Method to save the model parameters (A_inv and b for each arm)
    def save(self):
        # Support both linear and neural bandit persistence
        if self.bandit_type == "neural_linucb":
            # delegate save to neural implementation
            self.bandit.save(self.model_path)
            return
        temp_path = self.model_path + ".tmp.npz"                        # Save to as temporary file first to avoid corruption of the model original file
        np.savez(temp_path, A_inv=self.bandit.A_inv, b=self.bandit.b)   # Save the model parameters (A_inv and b for each arm)
        os.replace(temp_path, self.model_path)                          # Replace original model file by the new one

    # Method to load the model parameters from file, if it exists, otherwise initialize a new model
    def load(self):
        if not os.path.exists(self.model_path):                         # If the model file does not exist, we initialize a new model with default parameters (see __init__ method)
            return
        try:
            # If neural bandit requested, delegate loading to its loader 
            if self.bandit_type == "neural_linucb":
                try:
                    self.bandit.load(self.model_path)
                    print(f"Loaded neural model: {self.model_path}")
                except Exception:
                    print("Failed to load neural model; continuing with fresh model.")
                return
            data = np.load(self.model_path)                             # Load the model parameters from the .npz file
            self.bandit.A_inv = list(data['A_inv'])                     # Set A_inv for each arm from the loaded data
            self.bandit.b = list(data['b'])                             # Set b for each arm from the loaded data                 
            print(f"Loaded model: {self.model_path}")   
        except Exception:                                               # If there is an error loading the model, initialization of a new model (__init__)
            print("Corrupted model file. Reinitializing.")

    # Method to evaluate a position using the chess engine, returning a WDL score (win/draw/loss) for the current player
    def evaluate(self, board, depth=8):
        info = self.engine.analyse(                                 # engine.analyse to trigger Stockfish evaluation of the position
            board,                                                  # Board is a chess.Board object representing the current position for all pieces and game state
            chess.engine.Limit(depth=depth)                         # Depth is the number of moves the engine will look ahead to evaluate the position. 8 for average good balance
        )
        score = info["score"].white()                               # Get the score from analysis. (positive = white advantage, negative = black advantage)
        
        # Ajuste le score pour le joueur actuel
        if board.turn == chess.BLACK:
            score = -score  # Inverse si c'est le tour des noirs
            
        if score.is_mate():
            mate_moves = score.mate()
            if board.turn == chess.BLACK:
                mate_moves = -mate_moves  # Ajuste aussi pour les mats
            eval_cp = 10000 if mate_moves > 0 else -10000
        else:
            eval_cp = score.score()                                 # If it's not a mate score, we take the centipawn evaluation directly (positive if white is better, negative if black is better)
        wdl = 1.0 / (1.0 + 10.0 ** (-eval_cp / 400.0))              # Convert centipawn score to WDL (Win/Draw/Loss) probability using a logistic function.
                                                                    #   - wdl = 1.0: player has a winning position
                                                                    #   - wdl = 0.0: player has a losing position
                                                                    #   - wdl = 0.5: Balanced position 
                                                                    # --> TODO : Explain each part of 1.0 / (1.0 + 10.0 ** (-eval_cp / 400.0))
        return wdl

    # Method to extract features from the board and clock for the context representation used by the bandit algorithm
    def extract_features(self, board, clock):
        legal_moves = len(list(board.legal_moves))                                  # Number of legal moves in the current position
        captures = len([m for m in board.legal_moves if board.is_capture(m)])       # Number of legal moves that are captures (eat an opponent piece)
        features = np.array([                                                       # Feature vector = context
            legal_moves / 50.0,                                                     # Current position, normalized by a constant (e.g., 50) to keep it in a reasonable range for the model
            clock.ratio(),                                                          # Remaining time as a ratio of the initial time (between 0 and 1)
            board.fullmove_number / 100.0,                                          # Move number in game, normalized by 100 to keep every parameter in same range
            material_balance(board) / 39.0,                                         # Difference in piece values between the two players, normalized by the maximum possible imbalance : 39, to keep it between -1 and 1
            int(is_endgame(board)),                                                 # 1 if is endgame, 0 else
            int(board.is_check()),                                                  # 1 if the current player is in check, 0 else                  
            captures / 10.0                                                         # Normalized number of captures
        ])      
        return features.reshape(-1, 1)                                              # Reshape into a column vector: Array -> Column vector (n_features x 1)

    # Method to compute the reward for a move based on the change in position evaluation before and after the move, and the time taken to play the move
    def compute_reward(self, board, move, elapsed):
        score_before = self.evaluate(board, depth=8)            # Evaluate the position before playing the move (Perspective: Joueur A)
        board.push(move)                                        # Play the move on the board     
        score_after = self.evaluate(board, depth=8)             # Evaluate the position after playing the move (Perspective: Joueur B)
        board.pop()                                             # Undo the move to restore the original position (for other move evaluations)                        
        delta_wdl = (1.0 - score_after) - score_before          # On ajuste car score_after est du point de vue de B. 1 - score_after redonne la probabilité pour A.
        quality = delta_wdl * 10.0                              # Normalize the reward by a factor 10 to give more weight to the move
        reward = (                              
            2.0 * quality                                       # 
            - 0.05 * elapsed                                    # To penalize long moves 
        )
        return reward

    # Let's play ! training=True for training mode, False if not
    def play(self, board, clock, training=True):
        x = self.extract_features(                              # Extract board and clock from ChessMab instance
            board,
            clock
        )
        arm = self.bandit.select_arm(x)                         # Select an arm using the bandit 
        legal_moves = len(list(board.legal_moves))              # Extract number of legal moves for the current position
        budget = self.time_manager.compute_time_budget(         # Compute the time budget for the selected arm 
            arm=arm,    
            remaining_time=clock.time_left,                     # Time left from the start of the game for the current player
            legal_moves=legal_moves,
            move_number=board.fullmove_number,                  # Move number in the game, init at 1 and incremented after each black move
            endgame=is_endgame(board)
        )
        start = time.time()                                     # Start time to measure elapsed time for playing the move      
        result = self.engine.play(                              # Play the move using the chess engine with the computed time budget
            board,                                              
            chess.engine.Limit(time=budget)
        )
        elapsed = time.time() - start                           # Elapsed time for playing the move 
        move = result.move                                      # Move played by the engine (chess.Move object)                            
        reward = 0.0                                            # Init reward to 0 for the move
        # ===[Training only : not done in Test mode because it's not a Reinforcement Learning approach]===
        if training:
            reward = self.compute_reward(                       # Compute the reward for actual board, move and elapsed time
                board,
                move,
                elapsed
            )
            if clock.time_left - elapsed <= 0.0:                
                reward -= 10.0                                  # Penalize heavily if we run out of time after playing the move                 
            self.bandit.update(                                 # Bandit model update with the observed reward for the selected arm and context
                arm,
                x,
                reward
            )
        # =====================
        clock.spend(elapsed)                                    # Update the clock by spending the elapsed time for playing the move                        
        return (move, arm, reward, elapsed, budget)             # Return informations for analysis (webUI, logs)
