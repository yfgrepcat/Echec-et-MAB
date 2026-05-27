import glob
import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent


def _resolve_log_files(log_file=None, log_pattern="logs/games_worker_*.jsonl", worker_id=None):
    if worker_id:
        candidate = BASE_DIR / "logs" / f"games_worker_{worker_id}.jsonl"
        return [candidate] if candidate.exists() else []

    if log_file:
        candidate = Path(log_file)
        if candidate.is_file():
            return [candidate.resolve()]
        if not candidate.is_absolute():
            project_candidate = BASE_DIR / candidate
            if project_candidate.is_file():
                return [project_candidate.resolve()]
        raise FileNotFoundError(f"Log file not found: {candidate}")

    pattern = Path(log_pattern)
    if not pattern.is_absolute():
        pattern = BASE_DIR / pattern
    return [Path(path) for path in sorted(glob.glob(str(pattern)))]


def _infer_bandit_type(path, row):
    if row.get("bandit_type"):
        return row["bandit_type"]
    name = Path(path).name
    if "_neural" in name:
        return "neural_linucb"
    return "basic_linucb"


def _result_score(result):
    if result == "1-0":
        return 1.0
    if result == "1/2-1/2":
        return 0.5
    if result == "0-1":
        return 0.0
    return float("nan")


def _per_game_summary(df):
    rows = []
    if "game" not in df.columns:
        return pd.DataFrame(rows)

    for game_id, group in df.groupby("game", sort=True):
        last = group.iloc[-1]
        rows.append(
            {
                "game": int(game_id),
                "mab_moves": int(len(group)),
                "mean_reward": float(group["reward"].mean()) if "reward" in group else pd.NA,
                "mean_move_reward": (
                    float(group["move_reward"].mean()) if "move_reward" in group else pd.NA
                ),
                "result": last.get("result", None),
                "score": _result_score(last.get("result", None)),
                "mab_flagged": bool(last.get("mab_flagged", False)),
                "opponent_flagged": bool(last.get("opponent_flagged", False)),
            }
        )
    return pd.DataFrame(rows)


def _training_health(per_game, window_games=10):
    if per_game.empty:
        return {}

    window = max(1, min(int(window_games), len(per_game)))
    early = per_game.head(window)
    recent = per_game.tail(window)

    def _rates(frame):
        total = len(frame) or 1
        wins = float((frame["result"] == "1-0").sum()) / total
        draws = float((frame["result"] == "1/2-1/2").sum()) / total
        losses = float((frame["result"] == "0-1").sum()) / total
        score = float(frame["score"].mean()) if "score" in frame else float("nan")
        reward = float(frame["mean_reward"].mean()) if "mean_reward" in frame else float("nan")
        move_reward = (
            float(frame["mean_move_reward"].mean())
            if "mean_move_reward" in frame and frame["mean_move_reward"].notna().any()
            else float("nan")
        )
        return {
            "win_rate": round(wins, 3),
            "draw_rate": round(draws, 3),
            "loss_rate": round(losses, 3),
            "score_rate": round(score, 3),
            "mean_reward": round(reward, 4),
            "mean_move_reward": round(move_reward, 4) if pd.notna(move_reward) else None,
        }

    early_rates = _rates(early)
    recent_rates = _rates(recent)
    delta_score = round(recent_rates["score_rate"] - early_rates["score_rate"], 3)
    delta_reward = round(recent_rates["mean_reward"] - early_rates["mean_reward"], 4)
    delta_win = round(recent_rates["win_rate"] - early_rates["win_rate"], 3)
    delta_draw = round(recent_rates["draw_rate"] - early_rates["draw_rate"], 3)
    delta_loss = round(recent_rates["loss_rate"] - early_rates["loss_rate"], 3)

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
            "win_rate": delta_win,
            "draw_rate": delta_draw,
            "loss_rate": delta_loss,
        },
        "status": status,
        "interpretation": (
            "Recent results look better than the early games."
            if status == "improving"
            else "The signal is mixed; use more games or a longer window for a clearer verdict."
            if status == "mixed"
            else "Performance is roughly flat across the training window."
            if status == "stable"
            else "Recent results are weaker than the early training window."
        ),
    }


def load_training_logs(
    log_file=None,
    log_pattern="logs/games_worker_*.jsonl",
    worker_id=None,
    bandit_type=None,
):
    rows = []
    for path in _resolve_log_files(
        log_file=log_file,
        log_pattern=log_pattern,
        worker_id=worker_id,
    ):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row_bandit_type = _infer_bandit_type(path, row)
                if bandit_type and row_bandit_type != bandit_type:
                    continue
                row["bandit_type"] = row_bandit_type
                row["source_file"] = str(path)
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_training_logs(df):
    if df.empty:
        return {"stats": None}

    df = df.copy()
    for column in [
        "reward",
        "move_reward",
        "elapsed",
        "ply",
        "arm",
        "stockfish_level",
        "opponent_stockfish_level",
        "game",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    arm_counts = (
        df["arm"].dropna().astype(int).value_counts().sort_index().to_dict()
        if "arm" in df.columns else {}
    )

    if "game" in df.columns:
        reward_groups = ["source_file", "game"] if "source_file" in df.columns else ["game"]
        recent_rewards = (
            df.groupby(reward_groups, sort=False)["reward"]
            .mean()
            .tail(200)
            .round(6)
            .tolist()
        )
    else:
        recent_rewards = df["reward"].tail(200).round(6).tolist()

    level_column = (
        "opponent_stockfish_level"
        if "opponent_stockfish_level" in df.columns
        else "stockfish_level"
    )
    level_stats = {}
    if level_column in df.columns:
        for level, group in df.dropna(subset=[level_column]).groupby(level_column):
            if "game" in group.columns:
                rewards = group.groupby("game", sort=False)["reward"].mean().tail(200).round(6).tolist()
            else:
                rewards = group["reward"].tail(200).round(6).tolist()
            level_stats[str(int(level))] = {
                "avg_reward": round(float(group["reward"].mean()), 3),
                "rewards": rewards,
            }

    phase_stats = {}
    if "ply" in df.columns and "elapsed" in df.columns:
        df_phase = df.dropna(subset=["ply", "elapsed"]).copy()
        df_phase["move_number"] = (df_phase["ply"] // 2).astype(int)
        df_phase["phase_jeu"] = (df_phase["move_number"] // 5) * 5
        phase_data = df_phase.groupby("phase_jeu")["elapsed"].mean()
        phase_stats = {int(k): round(float(v), 2) for k, v in phase_data.items()}

    arm_time_stats = {}
    if "arm" in df.columns and "elapsed" in df.columns:
        arm_time_df = df.dropna(subset=["arm", "elapsed"])
        if not arm_time_df.empty:
            arm_time_data = arm_time_df.groupby("arm")["elapsed"].mean()
            arm_time_stats = {str(int(k)): round(float(v), 2) for k, v in arm_time_data.items()}

    per_game = _per_game_summary(df)
    outcome_counts = {}
    outcome_rates = {}
    training_health = {}
    if not per_game.empty:
        counts = per_game["result"].value_counts(dropna=False)
        wins = int(counts.get("1-0", 0))
        draws = int(counts.get("1/2-1/2", 0))
        losses = int(counts.get("0-1", 0))
        total = max(1, int(len(per_game)))
        outcome_counts = {"wins": wins, "draws": draws, "losses": losses}
        outcome_rates = {
            "win_rate": round(wins / total, 3),
            "draw_rate": round(draws / total, 3),
            "loss_rate": round(losses / total, 3),
            "score_rate": round(float(per_game["score"].mean()), 3),
        }
        training_health = _training_health(per_game, window_games=min(10, total))

    stats = {
        "rows": int(len(df)),
        "games": int(df["game"].nunique()) if "game" in df.columns else 0,
        "bandit_types": sorted(df["bandit_type"].dropna().unique().tolist()),
        "mean_reward": round(float(df["reward"].mean()), 4) if "reward" in df.columns else None,
        "mean_move_reward": (
            round(float(df["move_reward"].mean()), 4)
            if "move_reward" in df.columns and df["move_reward"].notna().any()
            else None
        ),
    }
    return {
        "stats": stats,
        "arm_counts": arm_counts,
        "recent_rewards": recent_rewards,
        "level_stats": level_stats,
        "phase_stats": phase_stats,
        "arm_time_stats": arm_time_stats,
        "outcome_counts": outcome_counts,
        "outcome_rates": outcome_rates,
        "training_health": training_health,
    }


def load_benchmark_results(bandit_type, csv_path=None):
    path = Path(csv_path) if csv_path else BASE_DIR / "logs" / f"benchmark_results_{bandit_type}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
