import chess
import chess.engine
import argparse
import pandas as pd
import sys
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from mab_agent import ChessMAB
from utils.time_manager import Clock
from utils.opening_book import load_openings, apply_random_opening

ENGINE_PATH = shutil.which("stockfish") or str(BASE_DIR / "bin" / "stockfish")


def run_benchmark(model_path, bandit_type, games_per_level, levels, time_control, use_openings):
    results = []
    openings = load_openings() if use_openings else []

    for level in levels:

        print(f"\n=== Benchmark vs SF Level {level} [{bandit_type}] ===")

        engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
        engine.configure({"Skill Level": level})

        wins = 0
        losses = 0
        draws = 0

        try:
            for game in range(games_per_level):

                print(f"Game {game + 1}/{games_per_level}")

                board = chess.Board()
                if use_openings:
                    board = apply_random_opening(board, openings)

                mab = ChessMAB(
                    engine,
                    model_path=model_path,
                    bandit_type=bandit_type,
                )

                mab_clock = Clock(time_control)

                while not board.is_game_over():

                    if board.turn == chess.WHITE:
                        move, *_ = mab.play(
                            board,
                            mab_clock,
                            training=False,
                        )
                    else:
                        result = engine.play(
                            board,
                            chess.engine.Limit(depth=6)
                        )
                        move = result.move

                    board.push(move)

                    if mab_clock.flag():
                        print("MAB flagged.")
                        break

                result = board.result()
                print("Result:", result)

                if result == "1-0":
                    wins += 1
                elif result == "0-1":
                    losses += 1
                else:
                    draws += 1

            results.append({
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
            engine.quit()

    print("\n=== FINAL RESULTS ===\n")
    print(pd.DataFrame(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(BASE_DIR / "models" / "final_model.npz"))
    parser.add_argument("--bandit-type", default="basic_linucb", choices=["basic_linucb", "neural_linucb"])
    parser.add_argument("--games-per-level", type=int, default=5)
    parser.add_argument("--levels", nargs="*", type=int, default=[1, 5, 10, 15, 20])
    parser.add_argument("--time-control", type=int, default=60)
    parser.add_argument("--no-openings", action="store_true")
    args = parser.parse_args()

    run_benchmark(
        model_path=args.model_path,
        bandit_type=args.bandit_type,
        games_per_level=args.games_per_level,
        levels=args.levels,
        time_control=args.time_control,
        use_openings=not args.no_openings,
    )
