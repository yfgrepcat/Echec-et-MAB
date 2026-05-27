import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Script used to analyse training logs and training progress.

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from experiments.reporting import load_training_logs, summarize_training_logs


ARM_COUNT = 4


def _resolve_output_dir(output_dir):
    if not output_dir:
        return None
    path = Path(output_dir)
    if not path.is_absolute():
        path = BASE_DIR / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_or_show(fig, output_dir, filename, show=True):
    fig.tight_layout()
    if output_dir:
        path = output_dir / filename
        fig.savefig(path, dpi=130, bbox_inches="tight")
        print(f"Saved plot: {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _prepare_logs(df):
    df = df.copy()
    numeric_columns = [
        "game",
        "ply",
        "arm",
        "reward",
        "move_reward",
        "elapsed",
        "budget",
        "outcome",
        "flag_penalty",
        "final_white_clock",
        "final_black_clock",
        "agent_stockfish_level",
        "opponent_stockfish_level",
        "stockfish_level",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "mab_flagged" in df.columns:
        df["mab_flagged"] = df["mab_flagged"].fillna(False).astype(bool)
    if "opponent_flagged" in df.columns:
        df["opponent_flagged"] = df["opponent_flagged"].fillna(False).astype(bool)
    return df


def _arm_share_by_game(df):
    counts = df.groupby(["game", "arm"]).size().unstack(fill_value=0)
    for arm in range(ARM_COUNT):
        if arm not in counts.columns:
            counts[arm] = 0
    counts = counts[sorted(counts.columns)]
    return counts.div(counts.sum(axis=1), axis=0).fillna(0.0)


def _normalized_entropy(row):
    values = np.asarray(row, dtype=float)
    values = values[values > 0]
    if len(values) == 0:
        return 0.0
    entropy = float(-(values * np.log(values)).sum() / math.log(ARM_COUNT))
    return max(0.0, min(1.0, entropy))


def _result_score(result):
    if result == "1-0":
        return 1.0
    if result == "1/2-1/2":
        return 0.5
    if result == "0-1":
        return 0.0
    return np.nan


def _per_game_summary(df):
    rows = []
    if "game" not in df.columns:
        return pd.DataFrame(rows)

    for game_id, group in df.groupby("game", sort=True):
        last = group.iloc[-1]
        row = {
            "game": int(game_id),
            "mab_moves": int(len(group)),
            "mean_reward": float(group["reward"].mean()) if "reward" in group else np.nan,
            "mean_move_reward": (
                float(group["move_reward"].mean()) if "move_reward" in group else np.nan
            ),
            "mean_elapsed": float(group["elapsed"].mean()) if "elapsed" in group else np.nan,
            "result": last.get("result", None),
            "score": _result_score(last.get("result", None)),
            "mab_flagged": bool(last.get("mab_flagged", False)),
            "opponent_flagged": bool(last.get("opponent_flagged", False)),
            "final_white_clock": float(last.get("final_white_clock", np.nan)),
            "final_black_clock": float(last.get("final_black_clock", np.nan)),
        }
        for arm in range(ARM_COUNT):
            row[f"arm{arm}_share"] = float((group["arm"] == arm).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _segment_summary(df, games_per_segment):
    per_game = _per_game_summary(df)
    if per_game.empty:
        return pd.DataFrame()

    rows = []
    min_game = int(per_game["game"].min())
    max_game = int(per_game["game"].max())
    for start in range(min_game, max_game + 1, games_per_segment):
        end = start + games_per_segment - 1
        games = per_game[(per_game["game"] >= start) & (per_game["game"] <= end)]
        if games.empty:
            continue
        segment_moves = df[df["game"].isin(games["game"])]
        total_games = max(1, int(len(games)))
        rows.append({
            "games": f"{start + 1}-{end + 1}",
            "logged_games": int(len(games)),
            "wins": int((games["result"] == "1-0").sum()),
            "draws": int((games["result"] == "1/2-1/2").sum()),
            "losses": int((games["result"] == "0-1").sum()),
            "win_rate": round(float((games["result"] == "1-0").sum()) / total_games, 3),
            "draw_rate": round(float((games["result"] == "1/2-1/2").sum()) / total_games, 3),
            "loss_rate": round(float((games["result"] == "0-1").sum()) / total_games, 3),
            "score_rate": round(float(games["score"].mean()), 3),
            "mab_flag_rate": round(float(games["mab_flagged"].mean()), 3),
            "opponent_flag_rate": round(float(games["opponent_flagged"].mean()), 3),
            "mean_mab_moves": round(float(games["mab_moves"].mean()), 1),
            "mean_final_white_clock": round(float(games["final_white_clock"].mean()), 2),
            "mean_reward": round(float(segment_moves["reward"].mean()), 4),
            "mean_move_reward": (
                round(float(segment_moves["move_reward"].mean()), 4)
                if "move_reward" in segment_moves.columns else np.nan
            ),
            "arm0": round(float((segment_moves["arm"] == 0).mean()), 3),
            "arm1": round(float((segment_moves["arm"] == 1).mean()), 3),
            "arm2": round(float((segment_moves["arm"] == 2).mean()), 3),
            "arm3": round(float((segment_moves["arm"] == 3).mean()), 3),
        })
    return pd.DataFrame(rows)


def _training_health_summary(per_game, window_games):
    if per_game.empty:
        return {}

    window = max(1, min(int(window_games), len(per_game)))
    early = per_game.head(window)
    recent = per_game.tail(window)

    def _rates(frame):
        total = max(1, len(frame))
        win_rate = float((frame["result"] == "1-0").sum()) / total
        draw_rate = float((frame["result"] == "1/2-1/2").sum()) / total
        loss_rate = float((frame["result"] == "0-1").sum()) / total
        score_rate = float(frame["score"].mean()) if "score" in frame else np.nan
        reward_rate = float(frame["mean_reward"].mean()) if "mean_reward" in frame else np.nan
        move_reward_rate = (
            float(frame["mean_move_reward"].mean())
            if "mean_move_reward" in frame and frame["mean_move_reward"].notna().any()
            else np.nan
        )
        return {
            "win_rate": round(win_rate, 3),
            "draw_rate": round(draw_rate, 3),
            "loss_rate": round(loss_rate, 3),
            "score_rate": round(score_rate, 3),
            "mean_reward": round(reward_rate, 4),
            "mean_move_reward": round(move_reward_rate, 4) if pd.notna(move_reward_rate) else None,
        }

    early_rates = _rates(early)
    recent_rates = _rates(recent)
    delta_score = round(recent_rates["score_rate"] - early_rates["score_rate"], 3)
    delta_reward = round(recent_rates["mean_reward"] - early_rates["mean_reward"], 4)

    if delta_score >= 0.05 and delta_reward >= 0:
        status = "improving"
    elif delta_score <= -0.05 and delta_reward <= 0:
        status = "declining"
    elif abs(delta_score) <= 0.03 and abs(delta_reward) <= 0.02:
        status = "stable"
    else:
        status = "mixed"

    return {
        "window_games": window,
        "early": early_rates,
        "recent": recent_rates,
        "delta": {
            "score_rate": delta_score,
            "mean_reward": delta_reward,
            "win_rate": round(recent_rates["win_rate"] - early_rates["win_rate"], 3),
            "draw_rate": round(recent_rates["draw_rate"] - early_rates["draw_rate"], 3),
            "loss_rate": round(recent_rates["loss_rate"] - early_rates["loss_rate"], 3),
        },
        "status": status,
    }


def _print_tables(df, games_per_segment, window_games):
    summary = summarize_training_logs(df)
    health = summary.get("training_health", {})
    print("\nLoaded rows:")
    print(len(df))
    print("\nShared summary:")
    print(summary["stats"])

    if summary.get("outcome_counts"):
        print("\nOverall outcomes:")
        print(summary["outcome_counts"])

    if summary.get("outcome_rates"):
        print("\nOverall outcome rates:")
        print(summary["outcome_rates"])

    print("\nAverage reward:")
    print(round(float(df["reward"].mean()), 5))

    if "move_reward" in df.columns:
        print("\nAverage move_reward:")
        print(round(float(df["move_reward"].mean()), 5))

    print("\nArm distribution:")
    print(df["arm"].value_counts(normalize=True).sort_index().round(3).to_string())

    print("\nAverage elapsed time by arm:")
    print(df.groupby("arm")["elapsed"].mean().round(3).to_string())

    per_game = _per_game_summary(df)
    if not per_game.empty:
        print("\nSegment summary:")
        segment = _segment_summary(df, games_per_segment)
        columns = [
            "games",
            "logged_games",
            "wins",
            "draws",
            "losses",
            "win_rate",
            "draw_rate",
            "loss_rate",
            "score_rate",
            "mean_reward",
        ]
        print(segment[columns].to_string(index=False))

    if health:
        print(f"\nTraining health ({health['window_games']} game window):")
        print(f"Status: {health['status']}")
        print(f"Early:  {health['early']}")
        print(f"Recent: {health['recent']}")
        print(f"Delta:  {health['delta']}")


def _plot_outcome_rates(df, output_dir, show, games_per_segment):
    segment = _segment_summary(df, games_per_segment)
    if segment.empty:
        return

    x = np.arange(len(segment))
    labels = segment["games"].tolist()
    fig, ax = plt.subplots(figsize=(13, 6))

    bottom = np.zeros(len(segment))
    palette = [
        ("loss_rate", "Loss rate", "#d62728"),
        ("draw_rate", "Draw rate", "#ff7f0e"),
        ("win_rate", "Win rate", "#2ca02c"),
    ]
    for column, label, color in palette:
        values = segment[column].astype(float).to_numpy()
        ax.bar(x, values, bottom=bottom, label=label, color=color, alpha=0.9)
        bottom += values

    ax.set_ylim(0, 1.05)
    ax.set_xlabel(f"Training segment ({games_per_segment} games per segment)")
    ax.set_ylabel("Outcome rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(x, segment["score_rate"], marker="o", color="#1f77b4", linewidth=2.5, label="Average score")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("Average score (win=1, draw=0.5, loss=0)")

    lines, labels_left = ax.get_legend_handles_labels()
    lines2, labels_right = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels_left + labels_right, loc="upper right")
    ax.set_title("Outcome rates by training segment")
    _save_or_show(fig, output_dir, "outcome_rates_by_segment.png", show=show)


def _plot_rewards(df, output_dir, show, window_games):
    per_game = _per_game_summary(df)
    if per_game.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    x = per_game["game"] + 1
    ax.plot(x, per_game["mean_reward"], marker="o", label="Per-game mean reward")
    if "mean_move_reward" in per_game.columns and per_game["mean_move_reward"].notna().any():
        ax.plot(x, per_game["mean_move_reward"], marker="o", label="Per-game mean move reward")
    ax.plot(
        x,
        per_game["mean_reward"].rolling(window_games, min_periods=1).mean(),
        linewidth=3,
        alpha=0.55,
        label=f"Reward rolling mean ({window_games} games)",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Training game")
    ax.set_ylabel("Reward")
    ax.set_title("Agent reward evolution")
    ax.legend()
    ax.grid(alpha=0.3)
    _save_or_show(fig, output_dir, "reward_by_game.png", show=show)


def _plot_training_health(df, output_dir, show, window_games):
    per_game = _per_game_summary(df)
    if per_game.empty:
        return

    health = _training_health_summary(per_game, window_games)
    x = per_game["game"] + 1
    rolling = per_game[["score", "mean_reward"]].rolling(window_games, min_periods=1).mean()
    if "mean_move_reward" in per_game.columns and per_game["mean_move_reward"].notna().any():
        rolling["mean_move_reward"] = per_game["mean_move_reward"].rolling(
            window_games,
            min_periods=1,
        ).mean()

    win_rate = per_game["result"].eq("1-0").rolling(window_games, min_periods=1).mean()
    draw_rate = per_game["result"].eq("1/2-1/2").rolling(window_games, min_periods=1).mean()
    loss_rate = per_game["result"].eq("0-1").rolling(window_games, min_periods=1).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        f"Training health overview - {health.get('status', 'unknown').title()}",
        y=0.98,
    )

    ax1.plot(x, win_rate, color="#2ca02c", label=f"Win rate ({window_games}-game rolling)")
    ax1.plot(x, draw_rate, color="#ff7f0e", label=f"Draw rate ({window_games}-game rolling)")
    ax1.plot(x, loss_rate, color="#d62728", label=f"Loss rate ({window_games}-game rolling)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_ylabel("Outcome rate")
    ax1.set_title("Rolling outcome rates")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="best")

    ax2.plot(x, rolling["mean_reward"], color="#1f77b4", linewidth=2.5, label="Reward rolling mean")
    if "mean_move_reward" in rolling.columns and rolling["mean_move_reward"].notna().any():
        ax2.plot(
            x,
            rolling["mean_move_reward"],
            color="#9467bd",
            linewidth=2,
            label="Move reward rolling mean",
        )
    ax2.set_ylabel("Reward")
    ax2.grid(alpha=0.3)
    ax2b = ax2.twinx()
    ax2b.plot(x, rolling["score"], color="#2ca02c", linestyle="--", linewidth=2, label="Score rolling mean")
    ax2b.set_ylabel("Average score")
    ax2.set_xlabel("Training game")

    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="best")
    ax2.set_title("Rolling reward and score")

    if health:
        fig.text(0.01, 0.01, f"{health['status'].title()}: {health.get('delta', {})}", fontsize=9)

    _save_or_show(fig, output_dir, "training_health.png", show=show)


def analyse_results(
    log_file=None,
    log_pattern="logs/games_worker_*.jsonl",
    worker_id=None,
    bandit_type=None,
    output_dir=None,
    show=True,
    games_per_segment=5,
    window_games=5,
):
    df = load_training_logs(
        log_file=log_file,
        log_pattern=log_pattern,
        worker_id=worker_id,
        bandit_type=bandit_type,
    )
    if df.empty:
        print("No data found in the selected log file(s).")
        return

    df = _prepare_logs(df)
    output_dir = _resolve_output_dir(output_dir)

    _print_tables(df, games_per_segment=games_per_segment, window_games=window_games)
    _plot_outcome_rates(df, output_dir, show, games_per_segment)
    _plot_rewards(df, output_dir, show, window_games)
    _plot_training_health(df, output_dir, show, window_games)


if __name__ == "__main__":
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
    parser.add_argument("--worker-id", help="Analyze one worker log by id.")
    parser.add_argument(
        "--bandit-type",
        choices=["basic_linucb", "neural_linucb"],
        help="Filter logs by bandit type.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory where PNG plots should be saved. Relative paths are under project/.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open Matplotlib windows; useful when saving plots.",
    )
    parser.add_argument(
        "--games-per-segment",
        type=int,
        default=10,
        help="Number of games per segment for outcome charts and summaries.",
    )
    parser.add_argument(
        "--window-games",
        type=int,
        default=10,
        help="Rolling window, in games, for reward and health curves.",
    )
    args = parser.parse_args()

    analyse_results(
        log_file=args.log_file,
        log_pattern=args.log_pattern,
        worker_id=args.worker_id,
        bandit_type=args.bandit_type,
        output_dir=args.output_dir,
        show=not args.no_show,
        games_per_segment=args.games_per_segment,
        window_games=args.window_games,
    )
