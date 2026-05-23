import argparse
import os
import json
import csv
import glob
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from training import run_training_session


BASE_DIR = Path(__file__).resolve().parent


def reset_benchmark_artifacts():
    """Remove previous benchmark outputs and worker models so runs are comparable."""
    patterns = [
        BASE_DIR / "logs" / "games_worker_*.jsonl",
        BASE_DIR / "logs" / "benchmark_results.csv",
        BASE_DIR / "models" / "worker_*.npz",
        BASE_DIR / "models" / "dry_run.npz",
    ]
    for pattern in patterns:
        for path in glob.glob(str(pattern)):
            try:
                os.remove(path)
            except IsADirectoryError:
                shutil.rmtree(path, ignore_errors=True)
            except FileNotFoundError:
                pass


def print_summary(summary_rows):
    by_bandit = defaultdict(lambda: {
        "runs": 0,
        "avg_reward_sum": 0.0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "arms": Counter(),
    })

    for row in summary_rows:
        entry = by_bandit[row["bandit_type"]]
        entry["runs"] += 1
        entry["avg_reward_sum"] += float(row["avg_reward"])
        results = row["results"]
        entry["wins"] += int(results.get("1-0", 0))
        entry["losses"] += int(results.get("0-1", 0))
        entry["draws"] += int(results.get("1/2-1/2", 0))
        entry["arms"].update(row["arm_counts"])

    print("\nBenchmark summary")
    print("-" * 72)
    for bandit_type, entry in by_bandit.items():
        runs = entry["runs"] or 1
        avg_reward = entry["avg_reward_sum"] / runs
        total_games = entry["wins"] + entry["losses"] + entry["draws"]
        winrate = entry["wins"] / total_games if total_games else 0.0
        arm_summary = ", ".join(f"{arm}:{count}" for arm, count in sorted(entry["arms"].items()))
        print(f"{bandit_type}: runs={runs}, games={total_games}, winrate={winrate:.3f}, avg_reward={avg_reward:.6f}")
        print(f"  results: wins={entry['wins']}, losses={entry['losses']}, draws={entry['draws']}")
        print(f"  arms: {arm_summary}")


def run_experiment(bandit_types, runs, games_per_run, device, force_cpu, simulate):
    os.makedirs(BASE_DIR / "logs", exist_ok=True)
    os.makedirs(BASE_DIR / "models", exist_ok=True)
    reset_benchmark_artifacts()
    run_id = 0
    summary_rows = []

    for btype in bandit_types:
        for r in range(runs):
            worker_id = run_id
            print(f"Starting run {run_id} for bandit {btype}")
            bandit_config = {"device": device, "force_cpu": force_cpu}
            # run training session (simulated)
            run_training_session(
                worker_id=worker_id,
                total_games=games_per_run,
                use_openings=False,
                use_random_positions=True,
                stockfish_level=1,
                time_control=60,
                bandit_type=btype,
                bandit_config=bandit_config,
                simulate=simulate,
            )

            # parse the worker log
            log_file = BASE_DIR / "logs" / f"games_worker_{worker_id}.jsonl"
            arms = Counter()
            rewards = []
            results = Counter()
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        if "arm" in obj and obj.get("arm") != "-":
                            arms[obj.get("arm")] += 1
                        if "reward" in obj:
                            try:
                                rewards.append(float(obj.get("reward", 0)))
                            except Exception:
                                pass
                        if "result" in obj and obj.get("result") is not None:
                            results[obj.get("result")] += 1

            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            row = {
                "run_id": run_id,
                "bandit_type": btype,
                "games": games_per_run,
                "avg_reward": avg_reward,
                "arm_counts": dict(arms),
                "results": dict(results),
            }
            summary_rows.append(row)
            run_id += 1
    # write CSV summary
    csv_path = BASE_DIR / "logs" / "benchmark_results.csv"
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["run_id", "bandit_type", "games", "avg_reward", "arm_counts", "results"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()})
    print(f"Benchmark complete. Results saved to {csv_path}")
    print_summary(summary_rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--games-per-run", type=int, default=5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--simulate", action="store_true", help="use DummyEngine instead of Stockfish")
    p.add_argument("--bandits", nargs="*", default=["basic_linucb", "neural_linucb"])
    args = p.parse_args()

    run_experiment(args.bandits, args.runs, args.games_per_run, args.device, bool(args.force_cpu), args.simulate)
