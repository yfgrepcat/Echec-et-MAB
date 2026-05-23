import chess
import chess.engine
import json
import os

from mab_agent import ChessMAB
from time_manager import Clock

from opening_book import (
    load_openings,
    apply_random_opening,
    load_random_middlegame
)


ENGINE_PATH = "stockfish"

TIME_CONTROL = 60

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)


def run_training_session(
    worker_id=0,
    total_games=100,
    use_openings=False,
    use_random_positions=True,
    stockfish_level=10,
    time_control=60,
    progress_callback=None
):

    log_file = f"logs/games_worker_{worker_id}.jsonl"

    engine = chess.engine.SimpleEngine.popen_uci(
        ENGINE_PATH
    )

    engine.configure({
        "Skill Level": stockfish_level
    })

    model_path = f"models/worker_{worker_id}.npz"

    mab = ChessMAB(
        engine,
        model_path=model_path
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


if __name__ == "__main__":

    run_training_session(
        worker_id=0,
        total_games=20,
        use_openings=False,
        use_random_positions=True,
        stockfish_level=10
    )
