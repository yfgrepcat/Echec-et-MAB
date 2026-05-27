"""Statistical diagnostic for the MAB reward function.

Answers two questions from a training log:

1. Is the reward function separable across arms? -- per-pair Cohen's d and
   the per-arm sample size needed to detect that effect at 80% power.
2. Is the per-move signal dominated by move QUALITY (delta_wdl) or by the
   TIME PENALTY? -- the same calculations on quality-only and penalty-only.

No new dependencies; reuses pandas/numpy/matplotlib. Recovers delta_wdl from
the logged reward by inverting compute_reward()'s formula.
"""

import argparse
import glob
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

# Mirrors compute_reward() in mab_agent.py (see lines 195-207):
#   reward = delta_wdl - time_penalty
#   time_penalty = TIME_PENALTY_COEF * min(elapsed / TIME_PENALTY_REF, 1.0)
TIME_PENALTY_COEF = 0.015  # Must match compute_reward() in mab_agent.py
TIME_PENALTY_REF = 1.5

# Critical values for a two-sided Welch t-test at alpha=0.05 with power=0.80.
# Sum z_{alpha/2} + z_{beta} = 1.96 + 0.8416 ~ 2.80; the per-group n then follows
# from Cohen's formula  n = 2 * ((z_{a/2} + z_b) / d)^2.
_Z_SUM_80_POWER = 2.80


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


def _load(log_file=None, log_pattern="logs/games_worker_*.jsonl"):
    rows = []
    for path in _resolve_log_files(log_file=log_file, log_pattern=log_pattern):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _decompose(df):
    # Recover time_penalty and delta_wdl from logged reward + elapsed.
    # New logs keep the dense per-move component in move_reward; use it so the
    # spread outcome bonus and flag penalty do not pollute arm separability.
    # Filter out terminal-dominated rows (|reward| close to 1.0) so the decomposition
    # only sees the per-move signal -- terminal rewards make delta_wdl recovery
    # nonsensical for the last-of-game ply.
    df = df.copy()
    reward_col = "move_reward" if "move_reward" in df.columns else "reward"
    df["reward"] = df[reward_col]
    df["time_penalty"] = TIME_PENALTY_COEF * np.minimum(df["elapsed"] / TIME_PENALTY_REF, 1.0)
    df["delta_wdl"] = df["reward"] + df["time_penalty"]
    # Drop rows whose reward looks dominated by terminal +-1.0 (i.e. last ply of a game).
    # These show up with |delta_wdl| > 0.6 once the terminal is folded in.
    return df[df["delta_wdl"].abs() < 0.6].copy()


def _welch_summary(x, y):
    """Mean diff y - x, pooled std, Cohen's d (Hedges' correction omitted),
    required per-arm n for 80% power, two-sided alpha=0.05."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return {"diff": float("nan"), "pooled_std": float("nan"),
                "cohen_d": float("nan"), "n_per_arm_for_80pct_power": float("nan")}
    mx, my = x.mean(), y.mean()
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    # Pooled standard deviation (equal-sample size formula); good enough when nx ~ ny.
    pooled = math.sqrt(((nx - 1) * sx ** 2 + (ny - 1) * sy ** 2) / (nx + ny - 2))
    diff = my - mx
    d = abs(diff) / pooled if pooled > 0 else float("inf")
    if d == 0 or not math.isfinite(d):
        n_required = float("inf")
    else:
        n_required = 2.0 * (_Z_SUM_80_POWER / d) ** 2
    return {"diff": diff, "pooled_std": pooled, "cohen_d": d,
            "n_per_arm_for_80pct_power": n_required}


def _format_n(n):
    if not math.isfinite(n):
        return "inf"
    if n > 1e7:
        return f"{n:.2e}"
    return f"{n:,.0f}"


def _print_pair_table(label, df, value_col, baseline_arm=0):
    print(f"\n=== {label} ===")
    arms = sorted(df["arm"].unique())
    if baseline_arm not in arms:
        baseline_arm = arms[0]
    base = df[df["arm"] == baseline_arm][value_col]
    print(f"baseline: arm {baseline_arm}  (n={len(base)}, mean={base.mean():+.4f}, std={base.std(ddof=1):.4f})")
    print(f"{'arm':>4} | {'n':>5} | {'mean':>9} | {'diff':>9} | {'cohen_d':>8} | {'n_per_arm_80%':>14}")
    print("-" * 64)
    for arm in arms:
        if arm == baseline_arm:
            continue
        other = df[df["arm"] == arm][value_col]
        s = _welch_summary(base, other)
        print(f"{arm:>4} | {len(other):>5} | {other.mean():+.4f} | "
              f"{s['diff']:+.4f} | {s['cohen_d']:>8.3f} | {_format_n(s['n_per_arm_for_80pct_power']):>14}")


def _verdict(df, baseline_arm=0):
    # The arm with the highest mean NET reward is what a converged bandit would pick.
    grouped = df.groupby("arm").agg(
        mean_reward=("reward", "mean"),
        mean_quality=("delta_wdl", "mean"),
        mean_time_pen=("time_penalty", "mean"),
        n=("reward", "count"),
    )
    best_net = grouped["mean_reward"].idxmax()
    best_quality = grouped["mean_quality"].idxmax()
    print("\n=== VERDICT ===")
    print(grouped.round(4).to_string())
    print(f"\nBy NET reward:     bandit converges on arm {best_net}  (mean reward {grouped.loc[best_net, 'mean_reward']:+.4f})")
    print(f"By QUALITY only:   ideal arm is        arm {best_quality}  (mean delta_wdl {grouped.loc[best_quality, 'mean_quality']:+.4f})")
    if best_net != best_quality:
        time_cost = grouped.loc[best_quality, "mean_time_pen"] - grouped.loc[best_net, "mean_time_pen"]
        quality_gain = grouped.loc[best_quality, "mean_quality"] - grouped.loc[best_net, "mean_quality"]
        net_loss = quality_gain - time_cost
        print(f"\n>> MISMATCH: time penalty ({time_cost:+.4f}) outweighs quality gain ({quality_gain:+.4f}) "
              f"by {-net_loss:+.4f}/move.")
        print(">> Lower the time penalty coefficient or remove it to let the bandit prefer the higher-quality arm.")
    else:
        print("\n>> Net-reward optimum agrees with quality optimum -- the reward function is internally consistent.")


def _plot_distributions(df, savepath=None):
    arms = sorted(df["arm"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax, col, title in zip(
        axes,
        ["reward", "delta_wdl", "time_penalty"],
        ["Net reward (per move)", "Quality only (delta_wdl)", "Time penalty only"],
    ):
        data = [df[df["arm"] == a][col].values for a in arms]
        means = [d.mean() for d in data]
        sems = [d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else 0.0 for d in data]
        ax.bar(arms, means, yerr=[1.96 * s for s in sems], capsize=4, color="#4682b4")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(arms)
        ax.set_xlabel("arm")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("mean (95% CI)")
    plt.suptitle("Reward separability: net vs quality vs time penalty")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=110, bbox_inches="tight")
        print(f"\nPlot saved to {savepath}")
    plt.show()


def analyse(log_file=None, log_pattern="logs/games_worker_*.jsonl", savepath=None):
    raw = _load(log_file=log_file, log_pattern=log_pattern)
    if raw.empty:
        print("No data found.")
        return
    df = _decompose(raw)
    print(f"Loaded {len(raw)} rows total, {len(df)} non-terminal moves used for separability.")
    print(f"Games: {raw['game'].nunique()}  Arms observed: {sorted(df['arm'].unique())}")

    _print_pair_table("NET reward (arm i vs arm 0)", df, "reward")
    _print_pair_table("QUALITY only (delta_wdl, time penalty removed)", df, "delta_wdl")
    _print_pair_table("TIME PENALTY only", df, "time_penalty")

    _verdict(df)
    _plot_distributions(df, savepath=savepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Statistical reward-function separability diagnostic for the MAB chess agent."
    )
    parser.add_argument("--log-file", help="Single log file to analyse.")
    parser.add_argument("--log-pattern", default="logs/games_worker_*.jsonl",
                        help="Glob pattern used when --log-file is not provided.")
    parser.add_argument("--save", help="Optional path to save the diagnostic plot (PNG).")
    args = parser.parse_args()
    analyse(log_file=args.log_file, log_pattern=args.log_pattern, savepath=args.save)
