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

## Run

```bash
python project/gui/app.py
```