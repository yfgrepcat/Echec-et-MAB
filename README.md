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

## Run training final night 

### Basic 300 stockfish 10 contre stockfish 8

```bash
python project/experiments/training.py --bandit-type basic_linucb --total-games 300 --worker-id p300_sf10_vs_sf8_basic --agent-stockfish-level 10 --opponent-stockfish-level 8
```

### Basic 300 stockfish 10 contre stockfish 10

```bash
python project/experiments/training.py --bandit-type basic_linucb --total-games 300 --worker-id p300_sf10_vs_sf10_basic --agent-stockfish-level 10 --opponent-stockfish-level 10
```

### Basic 300 stockfish 10 contre stockfish 12

```bash
python project/experiments/training.py --bandit-type basic_linucb --total-games 300 --worker-id p300_sf10_vs_sf12_basic --agent-stockfish-level 10 --opponent-stockfish-level 12
```

### Neural 300 stockfish 10 contre stockfish 8

```bash
python project/experiments/training.py --bandit-type neural_linucb --total-games 300 --worker-id p300_sf10_vs_sf8_neural --agent-stockfish-level 10 --opponent-stockfish-level 8
```

### Neural 300 stockfish 10 contre stockfish 10

```bash
python project/experiments/training.py --bandit-type neural_linucb --total-games 300 --worker-id p300_sf10_vs_sf10_neural --agent-stockfish-level 10 --opponent-stockfish-level 10
```

### Neural 300 stockfish 10 contre stockfish 12

```bash
python project/experiments/training.py --bandit-type neural_linucb --total-games 300 --worker-id p300_sf10_vs_sf12_neural --agent-stockfish-level 10 --opponent-stockfish-level 12
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

```

### Neural LinUCB 

```bash
./.venv/bin/python project/experiments/analysis.py --worker-id p100_sf10_neural --bandit-type neural_linucb
```
