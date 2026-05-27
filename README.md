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

## Run training 

### Basic LinUCB : agent Stockfish niveau 10 vs adversaire niveau 10, 100 parties

```bash
python project/experiments/training.py --bandit-type basic_linucb --total-games 100 --worker-id p100_sf10_basic --agent-stockfish-level 10 --opponent-stockfish-level 10
```

### Neural LinUCB : agent Stockfish niveau 10 vs adversaire niveau 10, 100 parties

```bash
python project/experiments/training.py --bandit-type neural_linucb --total-games 100 --worker-id p100_sf10_neural --agent-stockfish-level 10 --opponent-stockfish-level 10
```

## Run Benchmark 

### Basic LinUCB

```bash
python project/experiments/benchmark.py --bandit-type basic_linucb --model-path project/models/worker_p100_sf10_basic.npz --agent-stockfish-level 10 --levels 5 8 10 12
```

### Neural LinUCB 

```bash
python project/experiments/benchmark.py --bandit-type neural_linucb --model-path project/models/worker_p100_sf10_neural.npz --agent-stockfish-level 10 --levels 5 8 10 12
```

## Run Analysis

### Basic LinUCB

```bash
./.venv/bin/python project/experiments/analysis.py --worker-id p100_sf10_basic --bandit-type basic_linucb
```

### Neural LinUCB 

```bash
./.venv/bin/python project/experiments/analysis.py --worker-id p100_sf10_neural --bandit-type neural_linucb
```
