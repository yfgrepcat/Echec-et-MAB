# Chessomatic

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

mkdir -p project/bin
tmpdir=$(mktemp -d)
wget -O "$tmpdir/stockfish.tar" https://github.com/official-stockfish/Stockfish/releases/download/sf_18/stockfish-ubuntu-x86-64.tar
tar -xf "$tmpdir/stockfish.tar" -C "$tmpdir"
mv "$tmpdir/stockfish/stockfish-ubuntu-x86-64" project/bin/stockfish
chmod +x project/bin/stockfish
rm -rf "$tmpdir"
```

## Run the app 

```bash
python project/gui/app.py
```

## Run the Workflow

### 1. Training (with Auto-Checkpointing)

Train the MAB agent. The system automatically creates `.npz` model checkpoints in `project/models/checkpoints/` every 10 games.

To name your training runs, use the `--worker-id` argument (default is `0`). This dynamically configures all output filenames:
* **Active Model**: `project/models/worker_{worker_id}.npz`
* **Checkpoints**: `project/models/checkpoints/{worker_id}_{game_id}.npz`
* **Logs**: `project/logs/games_worker_{worker_id}.jsonl`

**Basic LinUCB (100 games vs Stockfish level 5, custom worker ID):**
```bash
python project/experiments/training.py --bandit-type basic_linucb --worker-id basic_sf5_run --total-games 100 --stockfish-level 5 --time-control 30
```

**Neural LinUCB (100 games vs Stockfish level 10, custom worker ID):**
```bash
python project/experiments/training.py --bandit-type neural_linucb --worker-id neural_sf10_run --total-games 100 --stockfish-level 10 --time-control 30
```

### 2. Evaluating Checkpoints

Once training has generated checkpoints, evaluate them across multiple opponent difficulty levels to generate a learning curve.
This will evaluate checkpoints starting with a specific worker ID (default is `0`). Pass `--worker-id` if you used a custom one during training.
This will output results to `project/logs/eval_results.csv`.

```bash
python project/experiments/evaluate_checkpoints.py --worker-id basic_sf5_run --games 5 --time-control 30
```

### 3. Generating Analysis

Generate learning curves and plot the MAB's arm selection distribution using the newly collected benchmark data.

```bash
python project/experiments/analysis.py
```

### 4. Running a Specific Benchmark Match

If you want to run a one-off match to test a specific model without running the full checkpoint evaluation suite:

```bash
python project/experiments/benchmark.py --bandit-type basic_linucb --model-path project/models/checkpoints/0_10.npz --time-control 30
```
