import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from experiments.reporting import load_logs

def plot_learning_curves(bandit_type):
    eval_csv = BASE_DIR / "logs" / "eval_results.csv"
    if not eval_csv.exists():
        print(f"\nNo eval_results.csv found at {eval_csv}")
        return

    df_eval = pd.read_csv(eval_csv)
    df_eval = df_eval[df_eval["bandit_type"] == bandit_type]
    
    if df_eval.empty:
        print(f"\nNo evaluation data for bandit_type: {bandit_type}")
        return

    # 1. Winrate vs Training Games
    plt.figure(figsize=(12, 5))
    for level, group in df_eval.groupby("opponent_level"):
        plt.plot(group["training_games"], group["winrate"], marker="o", label=f"vs Level {level}")
    
    plt.title(f"Learning Curve: Winrate vs Training Games ({bandit_type})")
    plt.xlabel("Training Games")
    plt.ylabel("Winrate")
    plt.legend()
    plt.grid()
    plt.show()

    # 2. Plies-to-win vs training games for weaker opponents (e.g. < 10)
    weaker_df = df_eval[df_eval["opponent_level"] < 10]
    if not weaker_df.empty:
        plt.figure(figsize=(12, 5))
        for level, group in weaker_df.groupby("opponent_level"):
            plt.plot(group["training_games"], group["mean_plies_to_win"], marker="o", label=f"vs Level {level}")
        
        plt.title(f"Learning Curve: Plies-to-Win vs Training Games ({bandit_type})")
        plt.xlabel("Training Games")
        plt.ylabel("Mean Plies to Win")
        plt.legend()
        plt.grid()
        plt.show()


def analyse_results(bandit_type="basic_linucb", worker_id=None):
    # Plot learning curves from evaluations first
    plot_learning_curves(bandit_type)

    # Then plot diagnostics from logs
    df = load_logs(bandit_type=bandit_type, worker_id=worker_id)
    if df.empty:
        print("\nNo data found in the selected log file(s).")
        return

    print(f"\nLoaded {len(df)} rows for bandit type: {bandit_type}")

    print("\nArm distribution:")
    print(df["arm"].value_counts())

    print("\nAverage elapsed time by arm:")
    print(df.groupby("arm")["elapsed"].mean())

    if "outcome" in df.columns:
        print("\nOutcome distribution:")
        print(df["outcome"].value_counts())

    # Stacked area of arm-usage proportion by game
    if "game" in df.columns:
        arm_share = (
            df.groupby(["game", "arm"]).size()
              .unstack(fill_value=0)
        )
        arm_share = arm_share.div(arm_share.sum(axis=1), axis=0)
        plt.figure(figsize=(12, 5))
        arm_share.plot.area(ax=plt.gca(), stacked=True)
        plt.title(f"Arm Usage Share per Game ({bandit_type})")
        plt.xlabel("Game index")
        plt.ylabel("Share")
        plt.ylim(0, 1)
        plt.grid()
        plt.show()

    # Average thinking time by arm
    plt.figure(figsize=(8, 5))
    df.groupby("arm")["elapsed"] \
        .mean() \
        .plot(kind="bar")
    plt.title(f"Average Thinking Time by Arm ({bandit_type})")
    plt.xlabel("Arm")
    plt.ylabel("Seconds")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bandit-type", default="basic_linucb", choices=["basic_linucb", "neural_linucb"])
    parser.add_argument("--worker-id", help="Analyze a single worker's logs.")
    args = parser.parse_args()

    analyse_results(bandit_type=args.bandit_type, worker_id=args.worker_id)