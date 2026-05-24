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

### Basic LinUCB

```bash
python project/experiments/benchmark.py \
  --bandit-type basic_linucb \
  --model-path project/models/worker_0.pkl
```

### Neural LinUCB 

```bash
python project/experiments/training.py \
  --bandit-type neural_linucb \
  --worker-id run_neural_v1 \
  --total-games 100
```

## Run Benchmark 

### Basic LinUCB

```bash
python project/experiments/benchmark.py \
  --bandit-type basic_linucb \
  --model-path project/models/worker_0.pkl
```

### Neural LinUCB 

```bash
python project/experiments/benchmark.py \
  --bandit-type neural_linucb \
  --model-path project/models/worker_0.pkl
```