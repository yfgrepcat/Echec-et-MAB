import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

# Ensure local project imports work when executing from repository root.
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import chess
import chess.engine

from mab_agent import ChessMAB, sanitize_bandit_config
from utils.time_manager import Clock
from utils.opening_book import (
    load_openings,
    apply_random_opening,
    load_random_middlegame
)

# Adjust if necessary to point to your personnal .venv bin directory
ENGINE_PATH = shutil.which("stockfish") or str(BASE_DIR / "bin" / "stockfish")       # Path to Stockfish bin 
TIME_CONTROL = 60                                       # Seconds per player for sthe training games

# Directories needed for logs and models to run training and benchmarking
os.makedirs(BASE_DIR / "logs", exist_ok=True)
os.makedirs(BASE_DIR / "models", exist_ok=True)

# DummyEngine just for smoke test and when Stockfish is not installed
# Same method signature as chess.engine.SimpleEngine    
class DummyEngine:

    # Result class to mimic chess.engine.PlayResult for the play method
    class Result:
        def __init__(self, move): self.move = move                              # The move to play  
    
    # Constructor for DummyEngine
    def __init__(self, skill_level=0): self.skill_level = skill_level           # The shill level used by model (ex. Stockfish level15)
    
    # Method needed to replicate Stockfish's configure method
    def configure(self, cfg): pass                                                                                                

    # Replicate Stockfish's play method
    def play(self, board, limit=None):
        moves = list(board.legal_moves)         # List of legal move from the board
        move = random.choice(moves)             # Pick a random legal move to play
        return DummyEngine.Result(move)         # Return the move to play

    # Fake analysis for the board, noting interesting
    def analyse(self, board, limit=None):
        class _Score:
            def white(self): return self
            def is_mate(self): return False
            def mate(self): return 0
            def score(self): return 0
        return {"score": _Score()}

    # Quit used when the game if finished
    def quit(self): pass

# Method to train a model : it will play `total_games` against Stockfish and log the results in a jsonl file
# Model trained is saved after each game to persist training. 
# Models are saved in `models/worker_{worker_id}.npz` and logs in `logs/games_worker_{worker_id}.jsonl` 
def run_training_session(
    worker_id="0",
    total_games=100,
    use_openings=False,
    use_random_positions=True,
    stockfish_level=3,
    time_control=60,
    bandit_type="basic_linucb",
    bandit_config=None,
    simulate=False,
    progress_callback=None
):

    worker_tag = str(worker_id)                                                     # Tag for the worker
    log_file = str(BASE_DIR / "logs" / f"games_worker_{worker_tag}.jsonl")          # Log file path

    # Try to start Stockfish engine. If not available and `simulate` is True,
    # fall back to a dummy engine that picks random legal moves. See DummyEngine class above
    # TODO : talk about it in the report
    agent_engine = None
    opponent_engine = None
    try:
        agent_engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
        agent_engine.configure({"Skill Level": 10})
        opponent_engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
        opponent_engine.configure({"Skill Level": stockfish_level})
    except FileNotFoundError:
        if simulate:
            print("Warning: 'stockfish' not found; using DummyEngine for simulation.")
            agent_engine = DummyEngine(skill_level=10)
            opponent_engine = DummyEngine(skill_level=stockfish_level)
        else:
            raise RuntimeError(
                "Stockfish engine not found in PATH. Install stockfish or run with --simulate to use a dummy engine."
            )

    # We need to sanitize the bandit to avoid errors when loading/saving the model
    # TODO : talk about this too on the report
    try:
        bandit_config = sanitize_bandit_config(bandit_config)                       # See method in mab_agent
    except ValueError as e:
        raise RuntimeError(f"Invalid bandit_config: {e}") from e

    mab = ChessMAB(                                                                 # Instialize instance of ChessMAB with the engine, model path and bandit config
        agent_engine,
        model_path=model_path,
        bandit_config=bandit_config,
        bandit_type=bandit_type,
    )

    openings = []                                                                   # Empty opening moves by default, will be loaded if use_openings is True
    try:
        for game_id in range(total_games):

            # Choose either we start training from initial position or middle game position (if True)
            if use_random_positions: board = load_random_middlegame()               # Start from a middle game position
            else:   
                board = chess.Board()                                               # Start from the initial position
                if use_openings:                                                    # If the user choosed to use openings
                    openings = load_openings()                                      # Load openings moves from book
                    board = apply_random_opening(                                   # Choose a moove and play it. See Method in utils/opening_book.py
                        board,
                        openings
                    )

            # Init clocks for both sides.
            mab_clock = Clock(time_control)
            opponent_clock = Clock(time_control)

            print(f"[Worker {worker_tag}]" 
                  f"Game {game_id + 1}/{total_games}")

            # Let's play !
            # Main game loop.
            # We buffer the history to apply the retroactive terminal reward
            # at the end of the game without double-counting context matrices.
            game_history = []
            pending = None

            def _flush(flagged=False):
                nonlocal pending
                if pending is None:
                    return
                arm_p, x_p, reward_p, log_p = pending
                
                # Update mid-game! The bandit learns from the immediate per-move signal.
                mab.bandit.update(arm_p, x_p, reward_p)
                
                # Save to history so we can apply terminal reward later
                game_history.append((arm_p, x_p, reward_p, log_p))
                pending = None

            def _end_game(flagged=False):
                if not game_history:
                    return

                total_moves = len(game_history)
                is_mab_flag = flagged and mab_clock.flag()
                is_opp_flag = flagged and opponent_clock.flag()
                
                # Small, spread game-outcome term and flag penalty
                outcome_sign = 0.0
                if is_mab_flag:
                    outcome_sign = -1.0
                elif is_opp_flag:
                    outcome_sign = 1.0
                elif board.is_game_over(claim_draw=True):
                    res = board.result(claim_draw=True)
                    if res == "1-0":
                        outcome_sign = 1.0
                    elif res == "0-1":
                        outcome_sign = -1.0
                
                spread_outcome = 0.0
                if outcome_sign != 0.0 and total_moves > 0:
                    mean_abs_reward = sum(abs(item[2]) for item in game_history) / total_moves
                    # The value added to each past move is a fraction (e.g., 50%) of the 
                    # average reward magnitude of the game. This ensures the episodic signal 
                    # never accidentally dwarfs the per-move learning signals.
                    spread_outcome = outcome_sign * (0.5 * mean_abs_reward)

                for arm_p, x_p, imm_reward, log_p in game_history:
                    if spread_outcome != 0.0:
                        mab.add_retroactive_reward(arm_p, x_p, spread_outcome)
                        
                    final_reward = imm_reward + spread_outcome
                    log_p["reward"] = final_reward
                    
                    if flagged:
                        log_p["outcome"] = "flagged"
                    elif board.is_game_over(claim_draw=True):
                        log_p["outcome"] = board.result(claim_draw=True)
                    else:
                        log_p["outcome"] = "unknown"
                        
                    with open(log_file, "a") as f:
                        json.dump(log_p, f)
                        f.write("\n")
                        f.flush()

                game_history.clear()

            while not board.is_game_over():
                if board.turn == chess.WHITE:                               # If White, it's turn of the mab agent
                    _flush()

                    move, arm, reward, elapsed, budget, x = mab.play(       # Play method of ChessMAB, see mab_agent.py for details
                        board,
                        mab_clock
                    )

                    log = {                                                 # Log entry; reward may be overwritten in _flush if a terminal bonus is added.
                        "worker": worker_tag,
                        "game": game_id,
                        "bandit_type": bandit_type,
                        "ply": board.ply(),
                        "move": board.san(move),
                        "arm": arm,
                        "reward": reward,
                        "elapsed": elapsed,
                        "budget": budget,
                        "legal_moves": len(
                            list(board.legal_moves)
                        ),
                        "clock": mab_clock.time_left,
                        "stockfish_level": stockfish_level
                    }

                    pending = (arm, x, reward, log)

                else:                                                       # It's Stockfish turn

                    start_opp = time.time()
                    result = opponent_engine.play(                                   # Stockfish, please play
                        board,
                        chess.engine.Limit(white_clock=mab_clock.time_left, black_clock=opponent_clock.time_left)
                    )

                    move = result.move                                      # Get the move played by Stockfish
                    elapsed_opp = time.time() - start_opp
                    opponent_clock.spend(elapsed_opp)

                board.push(move)                                            # Play the move, either from White or Black

                # Check if the clock has flagged for either player, if so, end the game
                if mab_clock.flag() or opponent_clock.flag():
                    print(
                        f"[Worker {worker_tag}] "
                        f"Flagged on time. MAB flagged: {mab_clock.flag()}, Opponent flagged: {opponent_clock.flag()}"
                    )
                    _flush(flagged=True)
                    _end_game(flagged=True)
                    break

                # If the game just ended (either side's move), flush so terminal reward lands on last White ply.
                if board.is_game_over(claim_draw=True):
                    _flush()
                    _end_game()

                # Update the progress of training
                if progress_callback:
                    progress_callback(game_id + 1, total_games, None)

            # Safety: flush any pending update if the loop exited without doing so.
            _flush()
            _end_game()

            # Save the current model after each game so training is persisted continuously.
            try:
                mab.save()                      # Save the model after each game to persist training
                # Checkpointing every 10 games
                if (game_id + 1) % 10 == 0:
                    checkpoint_dir = BASE_DIR / "models" / "checkpoints"
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path = str(checkpoint_dir / f"{worker_tag}_{game_id + 1}.npz")
                    mab_old_path = mab.model_path
                    mab.model_path = checkpoint_path
                    mab.save()
                    mab.model_path = mab_old_path
            except Exception as e:
                print(f"[Worker {worker_tag}] Warning: could not save model after game {game_id + 1}: {e}")

    finally:
        try:
            if agent_engine: agent_engine.quit()
            if opponent_engine: opponent_engine.quit()
        except Exception:
            pass

# Main method to parse arguments and start the training session
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train Chessomatic bandits against Stockfish.")
    parser.add_argument("--worker-id", type=str, default="0")
    parser.add_argument("--total-games", type=int, default=100)
    parser.add_argument("--use-openings", action="store_true")
    parser.add_argument("--no-random-positions", action="store_true", help="Start from the initial board instead of a random middlegame.")
    parser.add_argument("--stockfish-level", type=int, default=3)
    parser.add_argument("--time-control", type=int, default=60)
    parser.add_argument("--bandit-type", default="basic_linucb", choices=["basic_linucb", "neural_linucb"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--simulate", action="store_true", help="Use DummyEngine if Stockfish is unavailable.")
    args = parser.parse_args()

    # Bandit config extracted from arguments 
    bandit_config = {"device": args.device, "force_cpu": bool(args.force_cpu)}

    # Pump Pump Mab Muscles
    run_training_session(
        worker_id=args.worker_id,
        total_games=args.total_games,
        use_openings=args.use_openings,
        use_random_positions=not args.no_random_positions,
        stockfish_level=args.stockfish_level,
        time_control=args.time_control,
        bandit_type=args.bandit_type,
        bandit_config=bandit_config,
        simulate=args.simulate,
    )
