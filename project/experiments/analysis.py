import argparse
import json
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Script used to analyse in a terminal the results of benchmark and training sessions

BASE_DIR = Path(__file__).resolve().parent.parent

def _resolve_log_files(log_file=None, log_pattern="logs/games_worker_*.jsonl"):
    if log_file:
        candidate = Path(log_file)
        if candidate.is_file():
            return [str(candidate.resolve())]

        if not candidate.is_absolute():
            project_candidate = BASE_DIR / candidate
            if project_candidate.is_file():
                return [str(project_candidate.resolve())]

        raise FileNotFoundError(f"Log file not found: {candidate}")

    pattern = log_pattern
    if not Path(pattern).is_absolute():
        pattern = str(BASE_DIR / pattern)
    return sorted(glob.glob(pattern))


def analyse_results(log_file=None, log_pattern="logs/games_worker_*.jsonl"):
    # Load either one explicit log file or all matching worker logs.

    log_files = _resolve_log_files(log_file=log_file, log_pattern=log_pattern)
    if not log_files:
        print("No log files found.")
        return

    rows = []
    for log_file in log_files:
        with open(log_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))

    # Dataframe used to store and analyze the results
    df = pd.DataFrame(rows)
    if df.empty:
        print("No data found in the selected log file(s).")
        return

    print("\nLoaded rows:")
    print(len(df))
    print("\nColumns:")
    print(df.columns)
    print("\nHead:")
    print(df.head())

    # Statistics and visualizations of the results
    print("\nAverage reward:")
    print(df["reward"].mean())

    print("\nArm distribution:")
    print(df["arm"].value_counts())

    print("\nAverage elapsed time by arm:")
    print(df.groupby("arm")["elapsed"].mean())

    print("\nAverage reward by Stockfish level:")
    print(df.groupby("stockfish_level")["reward"].mean())

    # Raw reward for every move, without any aggregation or smoothing.
    # This shows the full per-move signal exactly as recorded in the logs.
    raw_reward = df["reward"].reset_index(drop=True)
    plt.figure(figsize=(12, 5))
    plt.scatter(raw_reward.index, raw_reward, s=10, alpha=0.5)
    plt.title("Raw Reward per Move")
    plt.xlabel("Move index")
    plt.ylabel("Reward")
    plt.grid()
    plt.show()

    # Graph of the rolling average reward over time (window 100 moves)
    plt.figure(figsize=(12, 6))
    rolling_reward = (
        df["reward"]
        .rolling(100)
        .mean()
    )
    plt.plot(rolling_reward)
    plt.title("Rolling Reward")
    plt.xlabel("Move")
    plt.ylabel("Reward")
    plt.grid()
    plt.show()

    # Graph of the distribution of arms used during the games
    plt.figure(figsize=(8, 5))
    df["arm"] \
        .value_counts() \
        .sort_index() \
        .plot(kind="bar")
    plt.title("Arm Usage")
    plt.xlabel("Arm")
    plt.ylabel("Count")
    plt.grid()
    plt.show()

    # Graph of the average thinking time by arm
    plt.figure(figsize=(8, 5))
    df.groupby("arm")["elapsed"] \
        .mean() \
        .plot(kind="bar")
    plt.title("Average Thinking Time by Arm")
    plt.xlabel("Arm")
    plt.ylabel("Seconds")
    plt.grid()
    plt.show()

    # Graph of average reward by Stockfish level
    plt.figure(figsize=(8, 5))
    df.groupby("stockfish_level")["reward"] \
        .mean() \
        .plot(kind="bar")
    plt.title("Average Reward vs Stockfish Level")
    plt.xlabel("Stockfish Level")
    plt.ylabel("Reward")
    plt.grid()
    plt.show()

    # Diagnostic plots for learning signal
    # The 100-move rolling mean above crosses game boundaries; the plots below avoid that
    # and surface the per-arm signal we actually need to verify the bandit is learning.

    if "game" in df.columns:
        # Mean reward per game: one point per game, immune to cross-game window artifacts.
        # Look for an upward trend over time -> the agent is actually learning to do better.
        plt.figure(figsize=(12, 5))
        df.groupby("game")["reward"].mean().plot(marker="o")
        plt.title("Mean Reward per Game")
        plt.xlabel("Game index")
        plt.ylabel("Mean reward")
        plt.grid()
        plt.show()

        # Stacked area of arm-usage proportion by game. Convergence on a particular arm
        # over time indicates the bandit is exploiting a learned preference.
        arm_share = (
            df.groupby(["game", "arm"]).size()
              .unstack(fill_value=0)
        )
        arm_share = arm_share.div(arm_share.sum(axis=1), axis=0)
        plt.figure(figsize=(12, 5))
        arm_share.plot.area(ax=plt.gca(), stacked=True)
        plt.title("Arm Usage Share per Game")
        plt.xlabel("Game index")
        plt.ylabel("Share")
        plt.ylim(0, 1)
        plt.grid()
        plt.show()

    # Per-arm rolling reward (overlaid): if curves overlap entirely, the reward does not
    # discriminate between arms and the bandit cannot learn from it (visual proof of the
    # arm-discrimination problem).
    plt.figure(figsize=(12, 5))
    for arm_id, group in df.groupby("arm"):
        group["reward"].rolling(50).mean().reset_index(drop=True).plot(label=f"arm {arm_id}")
    plt.title("Per-Arm Rolling Reward (window=50)")
    plt.xlabel("Move index within arm")
    plt.ylabel("Rolling mean reward")
    plt.legend()
    plt.grid()
    plt.show()

    # Reward distribution per arm. Overlapping boxes mean arms produce indistinguishable
    # reward distributions; well-separated boxes mean the bandit has a clear signal.
    plt.figure(figsize=(8, 5))
    df.boxplot(column="reward", by="arm")
    plt.title("Reward Distribution per Arm")
    plt.suptitle("")
    plt.xlabel("Arm")
    plt.ylabel("Reward")
    plt.grid()
    plt.show()

# Main method to parse arguments and start analysis
if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-file",
        help="Analyze a single log file instead of all worker logs.",
    )
    parser.add_argument(
        "--log-pattern",
        default="logs/games_worker_*.jsonl",
        help="Glob pattern used when --log-file is not provided.",
    )
    args = parser.parse_args()

    # Run analysis
    analyse_results(log_file=args.log_file, log_pattern=args.log_pattern)