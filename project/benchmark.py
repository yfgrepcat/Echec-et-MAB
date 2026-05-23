import chess
import chess.engine
import pandas as pd

from mab_agent import ChessMAB
from time_manager import Clock
from opening_book import load_openings, apply_random_opening

ENGINE_PATH = "stockfish"

LEVELS = [1, 5, 10, 15, 20]
RESULTS = []

openings = load_openings()

for level in LEVELS:

    print(f"\n=== Benchmark vs SF Level {level} ===")

    engine = chess.engine.SimpleEngine.popen_uci(
        ENGINE_PATH
    )

    engine.configure({
        "Skill Level": level
    })

    wins = 0
    losses = 0
    draws = 0

    for game in range(5):

        print(
            f"Game {game + 1}/5"
        )

        board = chess.Board()
        board = apply_random_opening(board, openings)

        mab = ChessMAB(
            engine,
            model_path="models/final_model.npz"
        )

        mab_clock = Clock(60)

        sf_clock = Clock(60)

        while not board.is_game_over():

            if board.turn == chess.WHITE:

                move, *_ = mab.play(
                    board,
                    mab_clock,
                    training=False
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

    RESULTS.append({
        "level": level,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "winrate": wins / 5.0
    })
    
    # Save incrementally
    pd.DataFrame(RESULTS).to_csv("logs/benchmark_results.csv", index=False)

    engine.quit()

print("\n=== FINAL RESULTS ===\n")
print(pd.DataFrame(RESULTS))
