# Neural LinUCB Integration Plan

## 1) Goal

Add a neural contextual bandit to the current chess agent while keeping the existing LinUCB workflow intact:
- same select then update loop
- same reward signal from game quality and time cost
- same training session and logging pipeline
- optional switch between linear and neural bandit

This plan is design-only and intentionally avoids production code.

## 2) Current Baseline (already in repo)

- Bandit decision is done in project/mab_agent.py
- Current policy is disjoint LinUCB with per-arm matrices A_inv and vectors b
- Context has 7 features extracted from board and clock state
- Online update happens after each move during training
- Training orchestration is in project/training.py and optionally through project/app.py endpoints

## 3) Recommended Architecture

### 3.1 Main Recommendation

Use a single shared neural encoder plus per-arm linear UCB head in latent space.

Why:
- shares information across arms
- keeps UCB uncertainty mechanism close to current LinUCB
- requires minimal conceptual change to existing loop

Concept:
- Encoder: z = f_theta(x), where x is 7-d context and z is latent vector
- Per arm a keep A_a^{-1} and b_a in latent space
- Linear posterior mean per arm: mu_a = theta_a^T z with theta_a = A_a^{-1} b_a
- UCB score: p_a = mu_a + alpha * sqrt(z^T A_a^{-1} z)

Selection:
- choose argmax_a p_a

Update:
- same Sherman-Morrison rank-1 update as today, but with z instead of x

### 3.2 Network Size

Start small for stability and speed:
- input: 7
- hidden: [64, 64]
- latent dimension: 16 or 32
- activation: ReLU
- optimizer: Adam, lr around 1e-3

### 3.3 Why not one network per arm first

Per-arm networks are heavier and harder to stabilize with limited data per arm.
Keep this as a second phase only if arms are strongly heterogeneous.

### 3.4 Context extension option

If needed later, augment context with:
- phase indicators (opening/middlegame/endgame)
- tactical pressure features
- recent time usage trend

Do this only after baseline neural integration works.

## 4) Device Strategy (no CUDA dependency required)

Default behavior:
- force CPU by default for reproducibility and portability

Optional acceleration:
- CUDA when available (NVIDIA)
- ROCm typically appears via torch CUDA interface on compatible AMD setups
- MPS for Apple Silicon (future-proof, even if not needed on Linux)

Practical rule:
- configuration flag device = auto | cpu | cuda | mps
- another flag force_cpu = true takes priority

## 5) File-Level Integration Plan

### 5.1 New file

Create project/neural_linucb.py with:
- NeuralLinUCB class
- internal encoder network
- methods: select_arm(x), update(arm, x, reward), save(path), load(path)

### 5.2 Existing file updates

project/mab_agent.py
- keep LinUCB class unchanged
- in ChessMAB, add bandit_type and bandit_config
- instantiate LinUCB or NeuralLinUCB depending on bandit_type
- adapt save/load path format for npz vs pt

project/training.py
- add bandit_type and bandit_config parameters to run_training_session
- choose model extension based on bandit type
- pass config into ChessMAB

project/app.py
- /api/start and /api/train accept optional bandit_type and bandit_config
- default remains linear LinUCB for backward compatibility

project/parallel_training.py
- set global BANDIT_TYPE and BANDIT_CONFIG for experiments

project/requirements.txt
- add torch

## 6) Training Protocol

Online loop remains identical:
1. extract context x
2. select arm from bandit
3. play move with budget policy
4. compute reward
5. update bandit with (arm, x, reward)

Neural-specific training policy:
- maintain replay buffer
- train encoder incrementally every N updates
- batch size small (16 to 64)
- optional gradient clipping for stability

Important:
- keep UCB matrices updated each step even if neural optimizer step is periodic

## 7) Evaluation Plan

Track at least:
- cumulative reward
- reward rolling mean
- arm distribution
- average time per arm
- win/draw/loss by stockfish level

Compare:
- baseline LinUCB vs NeuralLinUCB
- same random seeds and same game budgets

Decision rule:
- keep neural only if it improves reward trend and does not increase time losses

## 8) Risks and Mitigations

Risk: neural drift destabilizes UCB confidence
- Mitigation: small learning rate, periodic training, regularization

Risk: uncertainty term poorly scaled
- Mitigation: tune alpha separately for neural mode

Risk: model persistence mismatch
- Mitigation: strict file extension policy and versioned checkpoint metadata

Risk: training slowdown
- Mitigation: CPU-friendly network size and controlled update frequency

## 9) Step-by-Step Implementation Roadmap

Phase A: plumbing first
- Add bandit_type and config flow from app/training to ChessMAB
- Keep default linear mode untouched

Phase B: neural class skeleton
- Create NeuralLinUCB with API parity only
- Add save/load and basic tests for shape consistency

Phase C: action selection and update
- Implement latent UCB selection
- Implement Sherman-Morrison updates in latent space

Phase D: incremental neural learning
- Add replay buffer and mini-batch optimizer steps
- Add safe defaults and deterministic seeds

Phase E: experiment scripts
- Add one benchmark configuration for neural mode
- Compare with linear baseline

Phase F: tuning pass
- Tune alpha, latent dim, learning rate, train_every

## 10) First Implementation Slice (smallest useful increment)

Implement only:
- New NeuralLinUCB class with encoder + latent UCB (no advanced tricks)
- ChessMAB switch between linear and neural
- training.py parameter plumbing

Then run a short experiment and inspect logs before touching app endpoints.
