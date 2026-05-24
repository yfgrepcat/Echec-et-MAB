import random  # seed control for python random
from collections import deque  # fixed-size replay buffer
from typing import Optional

import numpy as np  # numerical arrays for A_inv and b
import torch  # PyTorch for encoder and training
import torch.nn as nn  # neural network building blocks


class RewardEncoder(nn.Module):

    def __init__(self, input_dim: int, hidden_sizes=(64, 64), repr_dim: int = 32):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))  # linear layer
            layers.append(nn.ReLU())  # activation
            in_dim = h
        layers.append(nn.Linear(in_dim, repr_dim))  # final projection to repr
        layers.append(nn.ReLU())  # non-linearity on representation
        self.net = nn.Sequential(*layers)  # stacked encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # return latent representation

# NeuralLinUCB is a contextual bandit algorithm that uses a neural network to learn a latent representation (lower-dimensional (reducted)) of the context, and then applies LinUCB in that latent space
# This neural network first turns the original chess context into a smaller learned representation and then the bandit uses a linear UCB on that learned representation instead of on the raw features
# The goal of this neuralLinUCB is to improve performance in complex contexts by learning a more efficient representation, leading to better exploration/exploitation decisions
# Implementation of this neural LinUCB is inspired by the paper "Neural Linear Bandits: Overcoming Catastrophic Forgetting through Experience Replay" (https://arxiv.org/pdf/1901.08612) and adapted to our chess context and constraints
# Copilot helped a lot to impement this class
class NeuralLinUCB:

    # Init method to create the neural LinUCB 
    def __init__(
        self,
        n_arms: int,                        # Number of arms (actions)
        n_features: int,                    # Input features 
        alpha: float = 1.5,                 # Exploration parameter (just like in basic LinUCB)
        hidden_sizes=(64, 64),              # Hidden layer size, default 2 layers of 64 units (neuron = unit)
        representation_dim: int = 32,       # Dimension of the latent representation (default 32)
        lr: float = 1e-3,                   # Learning rate for the encoder (Adam default: 0.001)
        ridge_lambda: float = 1.0,          # Regularization (ridge) for the linear part; affects init scale of A_inv
        batch_size: int = 32,               # Mini-batch size used when training the encoder from replay buffer
        train_every: int = 10,              # Perform a training step every N updates
        replay_size: int = 10000,           # Maximum size of the replay buffer
        seed: int = 42,                     # Random seed for reproducibility
        device: str = "auto",               # Device hint for torch: 'auto'|'cpu'|'cuda'|'mps'
        force_cpu: bool = False,            # If True, force CPU even when GPU is available
    ):

        self.n_arms = n_arms  # number of actions/arms
        self.n_features = n_features  # context dim
        self.alpha = alpha  # exploration coefficient for UCB
        self.repr_dim = representation_dim  # latent dimension
        self.batch_size = batch_size  # mini-batch size for encoder training
        self.train_every = max(1, int(train_every))  # frequency of training steps
        self.ridge_lambda = ridge_lambda  # regularization for A (implicit)

        self._rng = np.random.default_rng(seed)  # numpy RNG
        random.seed(seed)  # python RNG
        torch.manual_seed(seed)  # torch RNG

        self.device = self._resolve_device(device, force_cpu)  # compute torch device

        self.encoder = RewardEncoder(n_features, hidden_sizes, representation_dim).to(self.device)  # encoder model
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)  # optimizer
        self.loss_fn = nn.MSELoss()  # prediction loss

        init_scale = 1.0 / float(ridge_lambda)  # initial scaling for A_inv
        self.A_inv = [np.identity(representation_dim, dtype=np.float64) * init_scale for _ in range(n_arms)]  # per-arm A^-1
        self.b = [np.zeros((representation_dim, 1), dtype=np.float64) for _ in range(n_arms)]  # per-arm b

        self.buffer = deque(maxlen=int(replay_size))  # replay memory for incremental training
        self._update_steps = 0  # counter to schedule training

    @staticmethod
    def _resolve_device(device: str, force_cpu: bool):
        if force_cpu:
            return torch.device("cpu")  # force CPU when requested
        if device == "cpu":
            return torch.device("cpu")  # explicit CPU
        if device == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # prefer CUDA if available
        # auto fallback: prefer CUDA when available
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")  # otherwise CPU

    def _flatten_context(self, x) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)  # flatten to 1D float32
        if arr.shape[0] != self.n_features:
            raise ValueError(f"Context size mismatch: expected {self.n_features}, got {arr.shape[0]}")
        return arr

    def _encode_numpy(self, x1d: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(x1d).float().unsqueeze(0).to(self.device)  # to torch batch (1, n_features)
        with torch.no_grad():
            z = self.encoder(t).detach().cpu().numpy().reshape(-1, 1)  # get (repr_dim, 1) numpy
        return z.astype(np.float64)

    def select_arm(self, x) -> int:
        x1d = self._flatten_context(x)  # ensure correct shape
        z = self._encode_numpy(x1d)  # latent column vector (repr_dim, 1)

        scores = np.zeros(self.n_arms, dtype=np.float64)  # UCB scores per arm
        for arm in range(self.n_arms):
            theta = self.A_inv[arm] @ self.b[arm]  # posterior mean weights in latent space
            mean_reward = float((theta.T @ z).item())  # scalar mean
            uncertainty = np.sqrt((z.T @ self.A_inv[arm] @ z).item())  # scalar std
            scores[arm] = mean_reward + self.alpha * uncertainty  # UCB score
            
        max_score = np.max(scores)
        best_arms = [i for i, v in enumerate(scores) if v == max_score]
        import random
        return int(random.choice(best_arms))  # choose random arm among highest scores

    def update(self, arm: int, x, reward: float):
        x1d = self._flatten_context(x)  # prepare context
        z = self._encode_numpy(x1d)  # encode to latent

        # Sherman-Morrison rank-1 update for A_inv (efficient inverse update)
        num = (self.A_inv[arm] @ z) @ (z.T @ self.A_inv[arm])
        den = 1.0 + (z.T @ self.A_inv[arm] @ z).item()
        self.A_inv[arm] -= num / den  # update inverse covariance
        self.b[arm] += float(reward) * z  # update reward-weighted sum

        # push into replay buffer for encoder training
        self.buffer.append((x1d, int(arm), float(reward)))
        self._update_steps += 1

        # periodically perform a training step on the encoder
        if len(self.buffer) >= self.batch_size and (self._update_steps % self.train_every == 0):
            self._train_step()

    def _train_step(self):
        batch = random.sample(self.buffer, self.batch_size)  # sample mini-batch
        x_batch = np.stack([b[0] for b in batch]).astype(np.float32)  # (B, n_features)
        arm_batch = np.array([b[1] for b in batch], dtype=np.int64)  # (B,)
        reward_batch = np.array([b[2] for b in batch], dtype=np.float32)  # (B,)

        x_t = torch.from_numpy(x_batch).to(self.device)  # to device
        z_t = self.encoder(x_t)  # (B, repr_dim)

        preds = []
        for i in range(len(batch)):
            arm = int(arm_batch[i])
            theta = torch.from_numpy(self.A_inv[arm] @ self.b[arm]).float().squeeze(1).to(self.device)  # theta (repr_dim,)
            zi = z_t[i]  # latent for sample i
            pred = (zi @ theta).unsqueeze(0)  # scalar prediction
            preds.append(pred)

        pred_tensor = torch.cat(preds).view(-1)  # (B,)
        reward_t = torch.from_numpy(reward_batch).to(self.device)  # (B,)

        loss = self.loss_fn(pred_tensor, reward_t)  # MSE between predicted and observed reward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 5.0)  # clip grads for stability
        self.optimizer.step()

    def save(self, path: str):
        payload = {
            "encoder_state": self.encoder.state_dict(),  # network weights
            "optimizer_state": self.optimizer.state_dict(),  # optimizer state
            "A_inv": self.A_inv,  # per-arm inverse covariances
            "b": self.b,  # per-arm reward sums
            "n_arms": self.n_arms,
            "n_features": self.n_features,
            "repr_dim": self.repr_dim,
            "alpha": self.alpha,
        }
        torch.save(payload, path)  # atomic save handled by caller if desired

    def load(self, path: str):
        payload = torch.load(path, map_location=self.device)  # load checkpoint
        if "encoder_state" in payload:
            self.encoder.load_state_dict(payload["encoder_state"])  # restore weights
        if "optimizer_state" in payload:
            try:
                self.optimizer.load_state_dict(payload["optimizer_state"])  # restore optimizer
            except Exception:
                pass
        self.A_inv = [np.asarray(m, dtype=np.float64) for m in payload.get("A_inv", self.A_inv)]  # restore A_inv
        self.b = [np.asarray(v, dtype=np.float64) for v in payload.get("b", self.b)]  # restore b

