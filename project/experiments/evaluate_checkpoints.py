import argparse
import os
import shutil
import sys
import glob
from pathlib import Path
import re

import pandas as pd
import chess
import chess.engine

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from mab_agent import ChessMAB
from utils.time_manager import Clock
from utils.opening_book import load_openings, apply_random_opening

ENGINE_PATH = shutil.which("stockfish") or str(BASE_DIR / "bin" / "stockfish")

def evaluate_checkpoint(checkpoint_path, bandit_type, opponent_level, num_games, time_control, openings):
    wins = 0
    losses = 0
    draws = 0
    total_plies_win = 0
    total_clock_win = 0
    loss_on_time = 0

    agent_engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    agent_engine.configure({"Skill Level": 10})
    opponent_engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    opponent_engine.configure({"Skill Level": opponent_level})

    try:
        for _ in range(num_games):
            board = chess.Board()
            if openings:
                board = apply_random_opening(board, openings)

            mab = ChessMAB(
                agent_engine,
                model_path=str(checkpoint_path),
                bandit_type=bandit_type,
            )

            mab_clock = Clock(time_control)
            opponent_clock = Clock(time_control)

            while not board.is_game_over():
                if board.turn == chess.WHITE:
                    move, *_ = mab.play(board, mab_clock, training=False)
                else:
                    import time
                    start_opp = time.time()
                    result = opponent_engine.play(
                        board,
                        chess.engine.Limit(white_clock=mab_clock.time_left, black_clock=opponent_clock.time_left)
                    )
                    move = result.move
                    elapsed_opp = time.time() - start_opp
                    opponent_clock.spend(elapsed_opp)

                board.push(move)

                if mab_clock.flag() or opponent_clock.flag():
                    if mab_clock.flag():
                        loss_on_time += 1
                    break

            result = board.result(claim_draw=True)
            if mab_clock.flag():
                losses += 1
            elif opponent_clock.flag():
                wins += 1
                total_plies_win += board.ply()
                total_clock_win += (time_control - mab_clock.time_left)
            elif result == "1-0":
                wins += 1
                total_plies_win += board.ply()
                total_clock_win += (time_control - mab_clock.time_left)
            elif result == "0-1":
                losses += 1
            else:
                draws += 1
    finally:
        agent_engine.quit()
        opponent_engine.quit()

    winrate = wins / num_games if num_games > 0 else 0
    mean_plies_win = (total_plies_win / wins) if wins > 0 else 0
    mean_clock_win = (total_clock_win / wins) if wins > 0 else 0
    lot_rate = loss_on_time / num_games if num_games > 0 else 0

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "winrate": winrate,
        "mean_plies_to_win": mean_plies_win,
        "mean_clock_used_per_win": mean_clock_win,
        "loss_on_time_rate": lot_rate
    }

def run_evaluation(checkpoints_dir, output_csv, bandit_type, levels, games_per_eval, time_control, use_openings):
    checkpoints = glob.glob(str(Path(checkpoints_dir) / "*.npz"))
    
    parsed_checkpoints = []
    for cp in checkpoints:
        basename = os.path.basename(cp)
        match = re.match(r".*_(\d+)\.npz", basename)
        if match:
            parsed_checkpoints.append((int(match.group(1)), cp))
            
    parsed_checkpoints.sort(key=lambda x: x[0])
    
    if not parsed_checkpoints:
        print("No checkpoints found in", checkpoints_dir)
        return
        
    openings = load_openings() if use_openings else []

    all_results = []
    for games_trained, cp_path in parsed_checkpoints:
        print(f"\nEvaluating checkpoint after {games_trained} games: {cp_path}")
        for level in levels:
            print(f"  vs SF Level {level} ...")
            stats = evaluate_checkpoint(cp_path, bandit_type, level, games_per_eval, time_control, openings)
            result_row = {
                "training_games": games_trained,
                "opponent_level": level,
                "bandit_type": bandit_type,
                **stats
            }
            all_results.append(result_row)
            print(f"    Winrate: {stats['winrate']:.2f}, LOT Rate: {stats['loss_on_time_rate']:.2f}")

    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved evaluation results to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", default=str(BASE_DIR / "models" / "checkpoints"))
    parser.add_argument("--output", default=str(BASE_DIR / "logs" / "eval_results.csv"))
    parser.add_argument("--bandit-type", default="basic_linucb", choices=["basic_linucb", "neural_linucb"])
    parser.add_argument("--games", type=int, default=5, help="Eval games per checkpoint per opponent")
    parser.add_argument("--levels", nargs="+", type=int, default=[5, 8, 10, 12])
    parser.add_argument("--time-control", type=int, default=60)
    parser.add_argument("--no-openings", action="store_true")
    args = parser.parse_args()

    run_evaluation(
        args.checkpoints_dir,
        args.output,
        args.bandit_type,
        args.levels,
        args.games,
        args.time_control,
        not args.no_openings
    )
