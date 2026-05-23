import chess
import chess.engine
import json
import os
import random
from pathlib import Path
from mab_agent import ChessMAB, sanitize_bandit_config
from time_manager import Clock
from opening_book import (
    load_openings,
    apply_random_opening,
    load_random_middlegame
)

# Adjust if necessary to point to your personnal .venv bin directory
BASE_DIR = Path(__file__).resolve().parent              # Base directory of the project
ENGINE_PATH = str(BASE_DIR / "bin" / "stockfish")       # Path to Stockfish bin 
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

# 
def run_training_session(
    worker_id=0,
    total_games=100,
    use_openings=False,
    use_random_positions=True,
    stockfish_level=10,
    time_control=60,
    bandit_type="basic_linucb",
    bandit_config=None,
    simulate=False,
    progress_callback=None
):

    log_file = str(BASE_DIR / "logs" / f"games_worker_{worker_id}.jsonl")

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

    model_path = str(BASE_DIR / "models" / f"worker_{worker_id}.npz")

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
                f"[Worker {worker_id}] "
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
                        "worker": worker_id,
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
                        f"[Worker {worker_id}] "
                        f"Flagged on time."
                    )

                    break

            print(
                f"[Worker {worker_id}] "
                f"Result: {board.result()}"
            )
            # Append final game result to the log file for benchmarking/analysis
            with open(log_file, "a") as f:
                json.dump({"worker": worker_id, "game": game_id, "result": board.result()}, f)
                f.write("\n")
            if progress_callback:
                progress_callback(game_id + 1, total_games, board.result())

            mab.save()

    except KeyboardInterrupt:

        print(
            f"[Worker {worker_id}] "
            f"Training interrupted."
        )

        mab.save()

    engine.quit()


def _parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Run ChessMAB training or quick bandit dry-run.")
    p.add_argument("--bandit-type", default="basic_linucb", help="basic_linucb or neural_linucb")
    p.add_argument("--device", default="auto", help="device for neural bandit: auto|cpu|cuda|mps")
    p.add_argument("--force-cpu", action="store_true", help="force CPU even if CUDA available")
    p.add_argument("--simulate", action="store_true", help="run training with a dummy engine (no Stockfish required)")
    p.add_argument("--games", type=int, default=20, help="number of games to run in training mode")
    p.add_argument("--dry-run", action="store_true", help="quickly instantiate the agent and run a couple select/update ops (no engine plays)")
    p.add_argument("--time-control", type=int, default=TIME_CONTROL)
    p.add_argument("--stockfish-level", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    bandit_config = {"device": args.device, "force_cpu": bool(args.force_cpu)}
    try:
        bandit_config = sanitize_bandit_config(bandit_config)
    except ValueError as e:
        raise SystemExit(f"Invalid bandit_config: {e}")

    if args.dry_run:
        print("Dry-run: instantiating ChessMAB and running quick select/update checks")
        engine = None
        try:
            engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
            engine.configure({"Skill Level": args.stockfish_level})
        except FileNotFoundError:
            print("Warning: 'stockfish' not found in PATH; continuing without engine for dry-run.")

        model_path = str(BASE_DIR / "models" / "dry_run.npz")
        mab = ChessMAB(
            engine,
            model_path=model_path,
            bandit_config=bandit_config,
            bandit_type=args.bandit_type,
        )
        # quick smoke: extract features on a starting position and call select/update
        board = chess.Board()
        clock = Clock(args.time_control)
        x = mab.extract_features(board, clock)
        arm = mab.bandit.select_arm(x)
        print("selected arm", arm)
        mab.bandit.update(arm, x, reward=0.1)
        print("update ok")
        mab.save()
        mab.load()
        if engine is not None:
            engine.quit()
        print("Dry-run completed")
    else:
        run_training_session(
            worker_id=0,
            total_games=args.games,
            use_openings=False,
            use_random_positions=True,
            stockfish_level=args.stockfish_level,
            time_control=args.time_control,
            bandit_type=args.bandit_type,
            bandit_config=bandit_config,
            simulate=args.simulate,
        )
