import argparse
import shutil
import sys
from pathlib import Path

# Make sure python path is set to project root before local imports.
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import chess
import chess.engine
import pandas as pd

from mab_agent import ChessMAB
from utils.time_manager import Clock
from utils.opening_book import load_openings, apply_random_opening

ENGINE_PATH = shutil.which("stockfish") or str(BASE_DIR / "bin" / "stockfish")

# Class used to run a benchmark of MAB vs Stockfish for a range of levels
def run_benchmark(model_path, bandit_type, games_per_level, levels, time_control, use_openings):
    results = []                                                    # Where the results of the bechmark will be stored
    openings = load_openings() if use_openings else []              # Load openings if we want to use them

    for level in levels:
        print(f"\n=== Benchmark vs SF Level {level} [{bandit_type}] ===")       # For each level, we run games_per_level games against Stockfish at that level
        engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
        engine.configure({"Skill Level": level})                                # Set the level of Stockfish for this phase of the benchmark
        wins = 0
        losses = 0
        draws = 0

        try:
            for game in range(games_per_level):                                 # For each game, we create a new board and play until the end of the game
                                                                                # MAB to select moves for White and Stockfish to select moves for Black
                print(f"Game {game + 1}/{games_per_level}")
                board = chess.Board()
                if use_openings:
                    board = apply_random_opening(board, openings)

                mab = ChessMAB(                                                 # Initialize the MAB agent
                    engine,
                    model_path=model_path,
                    bandit_type=bandit_type,
                )

                mab_clock = Clock(time_control)                                 # Initialize the clock for Mab agent

                while not board.is_game_over():                                 # While game is not over

                    if board.turn == chess.WHITE:                               # Mab to play, so we get the other (we do not care about others returned values)
                        move, *_ = mab.play(
                            board,
                            mab_clock,
                            training=False,
                        )
                    else:
                        result = engine.play(                                   # Stockfish to play
                            board,
                            chess.engine.Limit(depth=6)
                        )
                        move = result.move                                      # Get the move
                    board.push(move)                                            # Play the move 

                    if mab_clock.flag():
                        print("MAB flagged.")
                        break

                result = board.result()                                         # Print result (1-0 White wins, 0-1 Black wins, or 1/2-1/2 Draw)
                print("Result:", result)

                if result == "1-0": wins += 1                                   # From the point of wiew of the MAB
                elif result == "0-1": losses += 1
                else: draws += 1

            results.append({                                                    # Appends the results of this level to the list and then to a csv file
                "bandit_type": bandit_type,
                "model_path": model_path,
                "level": level,
                "games": games_per_level,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "winrate": wins / float(games_per_level),
            })
            csv_path = BASE_DIR / "logs" / f"benchmark_results_{bandit_type}.csv"   
            pd.DataFrame(results).to_csv(str(csv_path), index=False)

        finally:
            engine.quit()                                                       # Quit properly Stockfish engine 

    print("\n=== FINAL RESULTS ===\n")
    print(pd.DataFrame(results))

# Main method to parse arguments and start the benchmark
if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(BASE_DIR / "models" / "final_model.npz"))
    parser.add_argument("--bandit-type", default="basic_linucb", choices=["basic_linucb", "neural_linucb"])
    parser.add_argument("--games-per-level", type=int, default=5)
    parser.add_argument("--levels", nargs="*", type=int, default=[1, 5, 10, 15, 20])
    parser.add_argument("--time-control", type=int, default=60)
    parser.add_argument("--no-openings", action="store_true")
    args = parser.parse_args()

    # Run benchmark 
    run_benchmark(
        model_path=args.model_path,
        bandit_type=args.bandit_type,
        games_per_level=args.games_per_level,
        levels=args.levels,
        time_control=args.time_control,
        use_openings=not args.no_openings,
    )
