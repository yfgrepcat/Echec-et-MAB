import argparse
import json
import os
import random
import shutil
import sys
import time
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
            def __neg__(self): return self
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
    stockfish_level=10,
    agent_stockfish_level=10,
    opponent_stockfish_level=None,
    time_control=60,
    bandit_type="basic_linucb",
    bandit_config=None,
    simulate=False,
    outcome_reward_scale=0.2,
    progress_callback=None
):

    worker_tag = str(worker_id)                                                     # Tag for the worker
    log_file = str(BASE_DIR / "logs" / f"games_worker_{worker_tag}.jsonl")          # Log file path
    model_path = str(BASE_DIR / "models" / f"worker_{worker_tag}.npz")              # Model file path
    opponent_stockfish_level = (
        stockfish_level if opponent_stockfish_level is None else opponent_stockfish_level
    )

    def _create_engine(level, label):
        try:
            engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
            engine.configure({"Skill Level": level})
            return engine
        except FileNotFoundError:
            if simulate:
                print(f"Warning: 'stockfish' not found; using DummyEngine for {label}.")
                return DummyEngine(skill_level=level)
            raise RuntimeError(
                "Stockfish engine not found in PATH. Install stockfish or run with --simulate to use a dummy engine."
            )

    def _limit_for_clocks(white_clock, black_clock):
        return chess.engine.Limit(
            white_clock=max(0.0, white_clock.time_left),
            black_clock=max(0.0, black_clock.time_left),
            white_inc=0.0,
            black_inc=0.0,
        )

    def _game_outcome(board, mab_flagged=False, opponent_flagged=False):
        if mab_flagged:
            return -1.0, "0-1"
        if opponent_flagged:
            return 1.0, "1-0"
        if not board.is_game_over(claim_draw=True):
            return 0.0, "*"
        result = board.result(claim_draw=True)
        if result == "1-0":
            return 1.0, result
        if result == "0-1":
            return -1.0, result
        return 0.0, result

    agent_engine = _create_engine(agent_stockfish_level, "agent")
    opponent_engine = _create_engine(opponent_stockfish_level, "opponent")

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

            # Both sides are on the same explicit clock; time pressure is now two-sided.
            mab_clock = Clock(time_control)
            opponent_clock = Clock(time_control)

            print(f"[Worker {worker_tag}]" 
                  f"Game {game_id + 1}/{total_games}")

            game_updates = []
            mab_flagged = False
            opponent_flagged = False

            while not board.is_game_over():
                if mab_clock.flag():
                    mab_flagged = True
                    break
                if opponent_clock.flag():
                    opponent_flagged = True
                    break

                if board.turn == chess.WHITE:                               # If White, it's turn of the mab agent
                    move, arm, reward, elapsed, budget, x = mab.play(       # Play method of ChessMAB, see mab_agent.py for details
                        board,
                        mab_clock
                    )

                    log = {                                                 # Log entry; reward may be overwritten in _flush if a terminal bonus is added.
                        "worker": worker_tag,
                        "game": game_id,
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
                        "white_clock": mab_clock.time_left,
                        "black_clock": opponent_clock.time_left,
                        "stockfish_level": opponent_stockfish_level,
                        "agent_stockfish_level": agent_stockfish_level,
                        "opponent_stockfish_level": opponent_stockfish_level,
                        "bandit_type": bandit_type,
                    }

                    game_updates.append((arm, x, reward, log))

                else:                                                       # It's Stockfish turn
                    start = time.time()
                    result = opponent_engine.play(                          # Stockfish, please play
                        board,
                        _limit_for_clocks(mab_clock, opponent_clock)
                    )
                    elapsed = time.time() - start
                    opponent_clock.spend(elapsed)

                    move = result.move                                      # Get the move played by Stockfish

                board.push(move)                                            # Play the move, either from White or Black

                # Check if the clock has flagged for either player, if so, end the game
                if mab_clock.flag():
                    print(
                        f"[Worker {worker_tag}] "
                        f"MAB flagged on time."
                    )
                    mab_flagged = True
                    break
                if opponent_clock.flag():
                    print(
                        f"[Worker {worker_tag}] "
                        f"Opponent flagged on time."
                    )
                    opponent_flagged = True
                    break

                # Update the progress of training
                if progress_callback:
                    progress_callback(game_id + 1, total_games, None)

            outcome, result_label = _game_outcome(
                board,
                mab_flagged=mab_flagged,
                opponent_flagged=opponent_flagged,
            )
            outcome_bonus = (
                outcome_reward_scale * outcome / len(game_updates)
                if game_updates else 0.0
            )

            with open(log_file, "a") as f:
                for index, (arm, x, move_reward, log) in enumerate(game_updates):
                    is_last_update = index == len(game_updates) - 1
                    flag_penalty = -1.0 if mab_flagged and is_last_update else 0.0
                    reward = move_reward + outcome_bonus + flag_penalty
                    log.update({
                        "move_reward": move_reward,
                        "outcome": outcome,
                        "outcome_bonus": outcome_bonus,
                        "flag_penalty": flag_penalty,
                        "reward": reward,
                        "result": result_label,
                        "mab_flagged": mab_flagged,
                        "opponent_flagged": opponent_flagged,
                        "final_white_clock": mab_clock.time_left,
                        "final_black_clock": opponent_clock.time_left,
                    })
                    mab.bandit.update(arm, x, reward)
                    json.dump(log, f)
                    f.write("\n")
                f.flush()

            # Save the current model after each game so training is persisted continuously.
            try:
                mab.save()
            except Exception as e:
                print(f"[Worker {worker_tag}] Warning: could not save model after game {game_id + 1}: {e}")

            if progress_callback:
                progress_callback(game_id + 1, total_games, result_label)

    finally:
        for engine in (agent_engine, opponent_engine):
            try:
                engine.quit()             # Quit the engine cleanly when training is done
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
    parser.add_argument("--stockfish-level", type=int, default=10, help="Backward-compatible alias for the opponent Stockfish level.")
    parser.add_argument("--agent-stockfish-level", type=int, default=10)
    parser.add_argument("--opponent-stockfish-level", type=int, default=None)
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
        agent_stockfish_level=args.agent_stockfish_level,
        opponent_stockfish_level=args.opponent_stockfish_level,
        time_control=args.time_control,
        bandit_type=args.bandit_type,
        bandit_config=bandit_config,
        simulate=args.simulate,
    )
