import argparse
import math
import shutil
import sys
import time
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
from experiments.training import DummyEngine

ENGINE_PATH = shutil.which("stockfish") or str(BASE_DIR / "bin" / "stockfish")


def _create_engine(skill_level, simulate=False):
    try:
        engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
        engine.configure({"Skill Level": skill_level})
        return engine
    except FileNotFoundError:
        if simulate:
            print(f"Warning: 'stockfish' not found; using DummyEngine at level {skill_level}.")
            return DummyEngine(skill_level=skill_level)
        raise


def _limit_for_clocks(white_clock, black_clock):
    return chess.engine.Limit(
        white_clock=max(0.0, white_clock.time_left),
        black_clock=max(0.0, black_clock.time_left),
        white_inc=0.0,
        black_inc=0.0,
    )


def _play_benchmark_game(
    mab,
    opponent_engine,
    time_control,
    openings=None,
):
    board = chess.Board()
    if openings:
        board = apply_random_opening(board, openings)

    mab_clock = Clock(time_control)
    opponent_clock = Clock(time_control)
    mab_flagged = False
    opponent_flagged = False

    while not board.is_game_over(claim_draw=True):
        if mab_clock.flag():
            mab_flagged = True
            break
        if opponent_clock.flag():
            opponent_flagged = True
            break

        if board.turn == chess.WHITE:
            move, *_ = mab.play(
                board,
                mab_clock,
                training=False,
            )
        else:
            start = time.time()
            result = opponent_engine.play(
                board,
                _limit_for_clocks(mab_clock, opponent_clock),
            )
            opponent_clock.spend(time.time() - start)
            move = result.move

        board.push(move)

        if mab_clock.flag():
            mab_flagged = True
            break
        if opponent_clock.flag():
            opponent_flagged = True
            break

    if mab_flagged:
        result_label = "0-1"
    elif opponent_flagged:
        result_label = "1-0"
    elif board.is_game_over(claim_draw=True):
        result_label = board.result(claim_draw=True)
    else:
        result_label = "*"

    mab_win = result_label == "1-0"
    return {
        "result": result_label,
        "plies": board.ply(),
        "mab_clock_used": time_control - mab_clock.time_left,
        "opponent_clock_used": time_control - opponent_clock.time_left,
        "mab_flagged": mab_flagged,
        "opponent_flagged": opponent_flagged,
        "plies_to_win": board.ply() if mab_win else math.nan,
        "clock_used_per_win": (time_control - mab_clock.time_left) if mab_win else math.nan,
    }


# Class used to run a benchmark of MAB vs Stockfish for a range of levels
def run_benchmark(
    model_path,
    bandit_type,
    games_per_level,
    levels,
    time_control,
    use_openings,
    agent_stockfish_level=10,
    output_path=None,
    write_results=True,
    simulate=False,
):
    results = []
    openings = load_openings() if use_openings else []

    output_path = output_path or BASE_DIR / "logs" / f"benchmark_results_{bandit_type}.csv"
    output_path = Path(output_path)
    if write_results:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    for level in levels:
        print(f"\n=== Benchmark agent SF {agent_stockfish_level} vs SF Level {level} [{bandit_type}] ===")
        agent_engine = _create_engine(agent_stockfish_level, simulate=simulate)
        opponent_engine = _create_engine(level, simulate=simulate)
        mab = ChessMAB(
            agent_engine,
            model_path=model_path,
            bandit_type=bandit_type,
        )

        game_rows = []
        try:
            for game in range(games_per_level):
                print(f"Game {game + 1}/{games_per_level}")
                row = _play_benchmark_game(
                    mab=mab,
                    opponent_engine=opponent_engine,
                    time_control=time_control,
                    openings=openings,
                )
                print("Result:", row["result"])
                game_rows.append(row)
        finally:
            agent_engine.quit()
            opponent_engine.quit()

        wins = sum(1 for row in game_rows if row["result"] == "1-0")
        losses = sum(1 for row in game_rows if row["result"] == "0-1")
        draws = sum(1 for row in game_rows if row["result"] == "1/2-1/2")
        loss_on_time = sum(1 for row in game_rows if row["mab_flagged"])
        wins_rows = [row for row in game_rows if row["result"] == "1-0"]

        results.append({
            "bandit_type": bandit_type,
            "model_path": str(model_path),
            "agent_stockfish_level": agent_stockfish_level,
            "opponent_level": level,
            "level": level,
            "games": games_per_level,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "winrate": wins / float(games_per_level),
            "mean_plies_to_win": (
                sum(row["plies_to_win"] for row in wins_rows) / len(wins_rows)
                if wins_rows else math.nan
            ),
            "mean_clock_used_per_win": (
                sum(row["clock_used_per_win"] for row in wins_rows) / len(wins_rows)
                if wins_rows else math.nan
            ),
            "loss_on_time_rate": loss_on_time / float(games_per_level),
        })
        if write_results:
            pd.DataFrame(results).to_csv(output_path, index=False)

    final = pd.DataFrame(results)
    print("\n=== FINAL RESULTS ===\n")
    print(final)
    return final


# Main method to parse arguments and start the benchmark
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(BASE_DIR / "models" / "final_model.npz"))
    parser.add_argument("--bandit-type", default="basic_linucb", choices=["basic_linucb", "neural_linucb"])
    parser.add_argument("--games-per-level", type=int, default=5)
    parser.add_argument("--levels", nargs="*", type=int, default=[5, 8, 10, 12])
    parser.add_argument("--time-control", type=int, default=60)
    parser.add_argument("--agent-stockfish-level", type=int, default=10)
    parser.add_argument("--no-openings", action="store_true")
    parser.add_argument("--simulate", action="store_true", help="Use DummyEngine if Stockfish is unavailable.")
    parser.add_argument("--output", help="CSV output path.")
    args = parser.parse_args()

    run_benchmark(
        model_path=args.model_path,
        bandit_type=args.bandit_type,
        games_per_level=args.games_per_level,
        levels=args.levels,
        time_control=args.time_control,
        use_openings=not args.no_openings,
        agent_stockfish_level=args.agent_stockfish_level,
        output_path=args.output,
        simulate=args.simulate,
    )
