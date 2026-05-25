import chess
import chess.engine
import json
import os
import random
import sys
import argparse
from pathlib import Path

# Adjust if necessary to point to your personnal .venv bin directory
BASE_DIR = Path(__file__).resolve().parent.parent              # Base directory of the project
import shutil
ENGINE_PATH = shutil.which("stockfish") or str(BASE_DIR / "bin" / "stockfish")       # Path to Stockfish bin 
TIME_CONTROL = 60                                       # Seconds per player for sthe training games

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from mab_agent import ChessMAB, sanitize_bandit_config
from utils.time_manager import Clock
from utils.opening_book import (
    load_openings,
    apply_random_opening,
    load_random_middlegame
)

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

# 
def run_training_session(
    worker_id="0",
    total_games=100,
    use_openings=False,
    use_random_positions=True,
    stockfish_level=15,
    time_control=60,
    bandit_type="basic_linucb",
    bandit_config=None,
    simulate=False,
    progress_callback=None
):

    worker_tag = str(worker_id)
    log_file = str(BASE_DIR / "logs" / f"games_worker_{worker_tag}.jsonl")

    # Try to start Stockfish engine. If not available and `simulate` is True,
    # fall back to a dummy engine that picks random legal moves. Otherwise,
    # raise a helpful error.
    engine = None
    try:
        engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
        engine.configure({"Skill Level": stockfish_level})
    except FileNotFoundError:
        if simulate:
            print("Warning: 'stockfish' not found; using DummyEngine for simulation.")
            engine = DummyEngine(skill_level=stockfish_level)
        else:
            raise RuntimeError(
                "Stockfish engine not found in PATH. Install stockfish or run with --simulate to use a dummy engine."
            )

    model_path = str(BASE_DIR / "models" / f"worker_{worker_tag}.npz")

    # validate/normalize bandit_config before creating the agent
    try:
        bandit_config = sanitize_bandit_config(bandit_config)
    except ValueError as e:
        raise RuntimeError(f"Invalid bandit_config: {e}") from e

    mab = ChessMAB(
        engine,
        model_path=model_path,
        bandit_config=bandit_config,
        bandit_type=bandit_type,
    )

    openings = []

    if use_openings:
        openings = load_openings()

    try:

        for game_id in range(total_games):

            # ---------------------------------
            # Initialisation position
            # ---------------------------------

            if use_random_positions:

                board = load_random_middlegame()

            else:

                board = chess.Board()

                if use_openings:
                    board = apply_random_opening(
                        board,
                        openings
                    )

            # ---------------------------------
            # Horloges
            # ---------------------------------

            mab_clock = Clock(time_control)
            sf_clock = Clock(time_control)

            print(
                f"[Worker {worker_tag}] "
                f"Game {game_id + 1}/{total_games}"
            )

            # ---------------------------------
            # Partie
            # ---------------------------------

            while not board.is_game_over():

                if board.turn == chess.WHITE:

                    move, arm, reward, elapsed, budget = mab.play(
                        board,
                        mab_clock
                    )

                    log = {
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
                        "stockfish_level": stockfish_level
                    }

                    with open(log_file, "a") as f:

                        json.dump(log, f)

                        f.write("\n")

                        f.flush()

                else:

                    result = engine.play(
                        board,
                        chess.engine.Limit(depth=10)
                    )

                    move = result.move

                board.push(move)

                # -----------------------------
                # Gestion du temps
                # -----------------------------

                if mab_clock.flag():

                    print(
                        f"[Worker {worker_tag}] "
                        f"Flagged on time."
                    )

                    break

                if progress_callback:
                    progress_callback(game_id + 1, total_games, None)

            # Save the current model after each game so training is persisted continuously.
            try:
                mab.save()
            except Exception as e:
                print(f"[Worker {worker_tag}] Warning: could not save model after game {game_id + 1}: {e}")

    finally:
        try:
            engine.quit()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Chessomatic bandits against Stockfish.")
    parser.add_argument("--worker-id", type=str, default="0")
    parser.add_argument("--total-games", type=int, default=100)
    parser.add_argument("--use-openings", action="store_true")
    parser.add_argument("--no-random-positions", action="store_true", help="Start from the initial board instead of a random middlegame.")
    parser.add_argument("--stockfish-level", type=int, default=10)
    parser.add_argument("--time-control", type=int, default=60)
    parser.add_argument("--bandit-type", default="basic_linucb", choices=["basic_linucb", "neural_linucb"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--simulate", action="store_true", help="Use DummyEngine if Stockfish is unavailable.")
    args = parser.parse_args()

    bandit_config = sanitize_bandit_config({"device": args.device, "force_cpu": bool(args.force_cpu)})

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
