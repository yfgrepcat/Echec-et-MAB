import json
import glob
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

def load_logs(bandit_type=None, worker_id=None):
    if worker_id:
        log_files = [str(BASE_DIR / "logs" / f"games_worker_{worker_id}.jsonl")]
    else:
        log_files = sorted(glob.glob(str(BASE_DIR / "logs" / "games_worker_*.jsonl")))

    rows = []
    for log_file in log_files:
        try:
            with open(log_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        row = json.loads(line)
                        if bandit_type is None or row.get("bandit_type", "basic_linucb") == bandit_type:
                            rows.append(row)
        except FileNotFoundError:
            pass
            
    if not rows:
        return pd.DataFrame()
        
    return pd.DataFrame(rows)
