# Echec et MAB ! (Chessomatic)


<p align="center">
	<img src="rsc/img/echec_et_mab.webp" alt="LinUCB vs Stockfish" width="220"/>
</p>

An approach to enhancing chess engine performance using Multi-Armed Bandits (MAB) in a time-constrained game of chess. This project implements both a basic LinUCB and a neural LinUCB agent, trained against Stockfish at various difficulty levels.

## Features used

1. **Number of legal moves:** Measures freedom of movement and positional complexity. A position with very few legal moves is often critical, whereas a more open position demands more evaluation.
2. **Remaining time ratio:** Indicates urgency relative to the opponent. Knowing that only $10\%$ of the time remains should push the agent to favor exploiting the fastest arms, regardless of board complexity.
3. **Game progression:** Evaluates the number of moves played to situate the maturity of the game. This helps identify the phase — opening, middlegame, or endgame — so as not to waste time on theoretical moves.
4. **Material balance:** Determines whether the position is asymmetric or whether the agent needs to force matters.
5. **Endgame phase:** Identifies the rarefaction of pieces, where time blunders are fatal. Precise calculation often becomes more important than intuition.
6. **King in check:** Signals an immediate crisis requiring a forced response.
7. **Possible captures:** Evaluates the immediate tactical tension on the board. The agent then immediately understands whether the slightest misstep would be fatal.


## Reward function

The reward function is the agent's only "compass": it mathematically translates what good behavior is — namely, finding excellent moves, but as quickly as possible. To model this problem, the reward is split into two components: a per-move reward and a terminal end-of-game reward.

$$R = \Delta\mathrm{WDL} - \mathrm{TimePenalty}$$

$$R = \Delta\mathrm{WDL} - 0.015 \times \min\left(\frac{\mathrm{ElapsedTime}}{1.5},\ 1\right)$$

Where WDL (Win/Draw/Loss) is the win probability computed by Stockfish before and after the move. Move quality is measured by **$\Delta\text{WDL}$** (positional improvement), then a time penalty is applied. The parameter choices are deliberate: **$\Delta\text{WDL}$** is bounded between -1 and 1, and the time penalty is capped via the `min` so that it does not dominate the entire signal and force the agent to constantly play too fast.



## Results

In one of our training runs, the basic LinUCB agent used a stockfish level 10 model and trained against a stockfish level 10 for 300 games. We can clearly see a net increase in both win-rate and mean game time.

<p align="center">
    <img src="rsc/img/stockfish_lvl10_vs_lvl10.webp" alt="LinUCB Win Rate" width="600"/>
</p>

Our models can be found under the [`project/models/`](https://github.com/yfgrepcat/Echec-et-MAB/tree/main/project/models) directory. All of the analysis of said models can be found under the [`project/reporting/`](https://github.com/yfgrepcat/Echec-et-MAB/tree/main/project/reporting) directory. The training logs are available under [`project/logs/`](https://github.com/yfgrepcat/Echec-et-MAB/tree/main/project/logs).

The training process allowed us to improve the MAB agent's performance against Stockfish, demonstrating the potential of MAB algorithms in optimizing decision-making under time constraints in chess. The results indicate that the agent can learn to select better moves over time and learns the crucial importance of time management in chess, as evidenced by the lowering of the mean game time.

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

Find below the commands to run training for different configurations. The trained models will be saved in `project/models/` and the logs in `project/logs/`.

### Basic LinUCB : agent Stockfish niveau 10 vs adversaire niveau 10, 100 parties

```bash
python project/experiments/training.py --bandit-type basic_linucb --total-games 100 --worker-id p100_sf10_basic --agent-stockfish-level 10 --opponent-stockfish-level 10
```

### Neural LinUCB : agent Stockfish niveau 10 vs adversaire niveau 10, 100 parties

```bash
python project/experiments/training.py --bandit-type neural_linucb --total-games 100 --worker-id p100_sf10_neural --agent-stockfish-level 10 --opponent-stockfish-level 10
```

## Final model training

Here are the commands that were run to train the final models that were used for the benchmark and analysis. Each command runs a training session of 300 games against the specified Stockfish level, with either the basic or neural LinUCB agent.

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

Benchmarking the trained models against Stockfish at levels 5, 8, 10, and 12. The benchmark will evaluate the performance of the trained agents against these different levels of Stockfish and generate logs for analysis.

### Basic LinUCB

```bash
python project/experiments/benchmark.py --bandit-type basic_linucb --model-path project/models/worker_p300_sf10_basic.npz --agent-stockfish-level 10 --levels 5 8 10 12
```

### Neural LinUCB 

```bash
python project/experiments/benchmark.py --bandit-type neural_linucb --model-path project/models/worker_p300_sf10_neural.npz --agent-stockfish-level 10 --levels 5 8 10 12
```

## Run Analysis (the graphs)

Analyzing the training logs and benchmark results to generate insights and visualizations about the performance of the trained agents. This includes plotting learning curves, win rates, and time management statistics.

### Basic LinUCB

```bash
./.venv/bin/python project/experiments/analysis.py --worker-id p300_sf10_basic --bandit-type basic_linucb
```

### Neural LinUCB 

```bash
./.venv/bin/python project/experiments/analysis.py --worker-id p300_sf10_neural --bandit-type neural_linucb
```
