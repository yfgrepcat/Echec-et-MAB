import argparse
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import matplotlib.pyplot as plt
import pandas as pd

from experiments.benchmark import run_benchmark


def _checkpoint_training_games(path, run_name=None):
    stem = Path(path).stem
    if run_name and stem.startswith(f"{run_name}_"):
        suffix = stem[len(run_name) + 1:]
    else:
        suffix = stem.rsplit("_", 1)[-1]
    match = re.search(r"(\d+)$", suffix)
    if not match:
        raise ValueError(f"Cannot infer training games from checkpoint name: {path}")
    return int(match.group(1))


def _resolve_checkpoints(run_name=None, checkpoint_glob=None):
    if checkpoint_glob:
        pattern = Path(checkpoint_glob)
        if not pattern.is_absolute():
            pattern = BASE_DIR / pattern
        paths = sorted(pattern.parent.glob(pattern.name))
    elif run_name:
        paths = sorted((BASE_DIR / "models" / "checkpoints").glob(f"{run_name}_*.npz"))
    else:
        raise ValueError("Provide --run or --checkpoint-glob.")

    checkpoints = []
    for path in paths:
        checkpoints.append((_checkpoint_training_games(path, run_name=run_name), path))
    return sorted(checkpoints, key=lambda item: item[0])


def _plot_learning_curves(df, run_name, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    winrate_path = output_dir / f"checkpoint_winrate_{run_name}.png"
    plt.figure(figsize=(10, 5))
    for level, group in df.groupby("opponent_level"):
        group = group.sort_values("training_games")
        plt.plot(group["training_games"], group["winrate"], marker="o", label=f"SF {level}")
    plt.title("Winrate vs training games")
    plt.xlabel("Training games")
    plt.ylabel("Winrate")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(winrate_path, dpi=120, bbox_inches="tight")
    plt.close()

    plies_path = output_dir / f"checkpoint_plies_to_win_{run_name}.png"
    weak = df[df["opponent_level"].isin([5, 8])]
    if not weak.empty:
        plt.figure(figsize=(10, 5))
        for level, group in weak.groupby("opponent_level"):
            group = group.sort_values("training_games")
            plt.plot(group["training_games"], group["mean_plies_to_win"], marker="o", label=f"SF {level}")
        plt.title("Plies to win vs training games")
        plt.xlabel("Training games")
        plt.ylabel("Mean plies to win")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(plies_path, dpi=120, bbox_inches="tight")
        plt.close()
    return winrate_path, plies_path


def evaluate_checkpoints(
    run_name=None,
    checkpoint_glob=None,
    bandit_type="basic_linucb",
    games_per_level=2,
    levels=None,
    time_control=60,
    agent_stockfish_level=10,
    use_openings=True,
    output=None,
    plot=True,
    simulate=False,
):
    levels = levels or [5, 8, 10, 12]
    checkpoints = _resolve_checkpoints(run_name=run_name, checkpoint_glob=checkpoint_glob)
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found.")

    label = run_name or "selected"
    rows = []
    for training_games, checkpoint_path in checkpoints:
        print(f"\n=== Evaluating checkpoint {checkpoint_path.name} ({training_games} games) ===")
        result = run_benchmark(
            model_path=str(checkpoint_path),
            bandit_type=bandit_type,
            games_per_level=games_per_level,
            levels=levels,
            time_control=time_control,
            use_openings=use_openings,
            agent_stockfish_level=agent_stockfish_level,
            write_results=False,
            simulate=simulate,
        )
        result = result.copy()
        result.insert(0, "training_games", training_games)
        result.insert(1, "checkpoint", str(checkpoint_path))
        rows.append(result)

    final = pd.concat(rows, ignore_index=True)
    output_path = Path(output) if output else BASE_DIR / "logs" / f"checkpoint_evaluation_{bandit_type}_{label}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_path, index=False)
    print(f"\nCheckpoint evaluation saved to {output_path}")

    if plot:
        winrate_path, plies_path = _plot_learning_curves(final, f"{bandit_type}_{label}", output_path.parent)
        print(f"Winrate plot saved to {winrate_path}")
        print(f"Plies-to-win plot saved to {plies_path}")

    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate checkpointed Chessomatic models.")
    parser.add_argument("--run", help="Run name used in models/checkpoints/{run}_{games}.npz.")
    parser.add_argument("--checkpoint-glob", help="Alternative checkpoint glob, relative to project/ if not absolute.")
    parser.add_argument("--bandit-type", default="basic_linucb", choices=["basic_linucb", "neural_linucb"])
    parser.add_argument("--games-per-level", type=int, default=2)
    parser.add_argument("--levels", nargs="*", type=int, default=[5, 8, 10, 12])
    parser.add_argument("--time-control", type=int, default=60)
    parser.add_argument("--agent-stockfish-level", type=int, default=10)
    parser.add_argument("--no-openings", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--simulate", action="store_true", help="Use DummyEngine if Stockfish is unavailable.")
    parser.add_argument("--output", help="CSV output path.")
    args = parser.parse_args()

    evaluate_checkpoints(
        run_name=args.run,
        checkpoint_glob=args.checkpoint_glob,
        bandit_type=args.bandit_type,
        games_per_level=args.games_per_level,
        levels=args.levels,
        time_control=args.time_control,
        agent_stockfish_level=args.agent_stockfish_level,
        use_openings=not args.no_openings,
        output=args.output,
        plot=not args.no_plots,
        simulate=args.simulate,
    )
