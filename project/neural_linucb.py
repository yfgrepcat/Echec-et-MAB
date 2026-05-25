import random  # seed control for python random
from collections import deque  # fixed-size replay buffer
from typing import Optional

import numpy as np  # numerical arrays for A_inv and b
import torch  # PyTorch for encoder and training
import torch.nn as nn  # neural network building blocks


# Rewardnetwork is a simple neural network that takes features of the chess context and output lattent reprensentation
class RewardNetwork(nn.Module):

    # Method to initialize the network with one hidden layer by default and ReLU activation 
    def __init__(self, input_dim: int, hidden_sizes=(32,), repr_dim: int = 16):
        super().__init__()                                  # To initialize the nn.Module parent class
        layers = []                                         # layers is a list that will contain the layers of the network, built sequentially
        in_dim = input_dim                                  # Number in of feature, the input dimension of first layer
        for h in hidden_sizes:                              # For each hidden layer, we add a linear layer followed by a ReLU activation
            layers.append(nn.Linear(in_dim, h))             
            layers.append(nn.ReLU())        
            in_dim = h                                      # Update in_dim for the next layer, the output size of the current layer
        layers.append(nn.Linear(in_dim, repr_dim))          # Final layer to output the latent reprensentaiton,
        layers.append(nn.ReLU())                            
        self.net = nn.Sequential(*layers)                   # Sequential permit to chain the layer toguether

    # Method to forward the input trough the network layer and get the latent representation
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# NeuralLinUCB is a contextual bandit algorithm that uses a neural network to learn a latent representation (lower-dimensional (reducted)) of the context, and then applies LinUCB in that latent space
# This neural network first turns the original chess context into a smaller learned representation and then the bandit uses a linear UCB on that learned representation instead of on the raw features
# The goal of this neuralLinUCB is to improve performance in complex contexts by learning a more efficient representation, leading to better exploration/exploitation decisions
# Implementation of this neural LinUCB is inspired by the paper "Neural Linear Bandits: Overcoming Catastrophic Forgetting through Experience Replay" (https://arxiv.org/pdf/1901.08612) and adapted to our chess context and constraints
# Copilot helped a lot to impement this class and to understand the maths
# Each neuron act like that y=activation(w⋅x+b), where w is the weight vector, x is the input vector, b is the bias, and activation is a function like ReLU (same used in the linUCB)
# The neural network is trained once in a while (train_every steps) using a buffer that stores past contexts, actions, and rewards. This is how the encoder learns
# NeuralLinUCB first maps the 7 chess context features to a learned latent vector,
# then LinUCB works on that latent vector instead of on the raw features.
# The latent size is a design choice: it does not come from the data automatically.
# This version is inspired by Neural Linear Bandits, but kept intentionally small
class NeuralLinUCB:

    # Init method to create the neural LinUCB.
    # The encoder takes 7 input features and outputs a latent vector of size representation_dim.
    def __init__(
        self,
        n_arms: int,                        # Number of arms (actions)
        n_features: int,                    # Input features 
        alpha: float = 1.5,                 # Exploration parameter (just like in basic LinUCB)
        hidden_sizes=(10,),                 # One hidden layer: 10 neurons, simple and enough for the current 7 features
        representation_dim: int = 16,       # Output size of the encoder; this is the latent context size used by LinUCB
        lr: float = 1e-3,                   # Learning rate for the encoder network
        ridge_lambda: float = 1.0,          # Regularization (ridge) for the linear part; affects init scale of A_inv
        batch_size: int = 32,               # Number of past training examples sampled at once from the replay buffer
        train_every: int = 5,              # Perform a training step every N updates
        replay_size: int = 10000,           # Maximum size of the replay buffer
        seed: int = 42,                     # Random seed for reproducibility
        device: str = "auto",               # Device hint for torch: 'auto'|'cpu'|'cuda'|'mps'
        force_cpu: bool = False,            # If True, force CPU even when GPU is available
    ):
        self.n_arms = n_arms 
        self.n_features = n_features
        self.alpha = alpha
        self.repr_dim = representation_dim
        self.batch_size = batch_size
        self.train_every = max(1, int(train_every))
        self.ridge_lambda = ridge_lambda
        self._rng = np.random.default_rng(seed)             # Seeds permit to have reproductible results
        random.seed(seed)                                   # Seed for python random, used in select_arm to break ties randomly
        torch.manual_seed(seed)                             # Initialization of weights in the encoder, deterministic with the same seed
        self.device = self._resolve_device(device, force_cpu)                                                   # (CPU/GPU)
        self.neural_network = RewardNetwork(n_features, hidden_sizes, representation_dim).to(self.device)       # Encoder is the neural network in charge of creating the latent representation
        self.optimizer = torch.optim.Adam(self.neural_network.parameters(), lr=lr)                              # Adam optimizer, is here to adapt weights for each encoder neuron during training
        self.loss_fn = nn.MSELoss()                                                                             # Prediction loss, using MSE (mean squared error) between predicted reward and observed reward during training of the encoder                                   
        init_scale = 1.0 / float(ridge_lambda)                                                                  
        self.A_inv = [np.identity(representation_dim, dtype=np.float64) * init_scale for _ in range(n_arms)]    # Initial A_a^-1 for each arm a (identity matrix) 
        self.b = [np.zeros((representation_dim, 1), dtype=np.float64) for _ in range(n_arms)]                   # Initial b_a for each arm a (zero vector) : vector reward
        self.buffer = deque(maxlen=int(replay_size))        # Replay memory of played actions for training
        self._update_steps = 0                              # Counter to schedule training, every train_every steps

    # Method to resolve on wich device to run the encoder
    @staticmethod
    def _resolve_device(device: str, force_cpu: bool):
        if force_cpu: return torch.device("cpu")                                                        # Force CPU when requested
        if device == "cpu": return torch.device("cpu")                                                  # Explicit CPU
        if device == "cuda": return torch.device("cuda" if torch.cuda.is_available() else "cpu")        # Prefer CUDA if available
        if torch.cuda.is_available(): return torch.device("cuda")
        return torch.device("cpu")                                                                      # Otherwise CPU

    # Helper method to flatten the 7 input features into a 1D vector. Prepare feature vector for encoder
    # Convern imput info a NumPy 1D array of shape (n_features,) with dtype float32
    def _flatten_context(self, x) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)       # Convert into NumPy array, force float32,flatten to 1D
        if arr.shape[0] != self.n_features:                     # Vector length need to be exactly n_features
            raise ValueError(f"Context size mismatch: expected {self.n_features}, got {arr.shape[0]}")
        return arr

    # Method to encode the flattened context into latent representation 
    def _encode_numpy(self, x1d: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(x1d).float().unsqueeze(0).to(self.device)      # PyToroch tensor shared in memory
        with torch.no_grad():                                               # No gradient needed for encoding during action selection
            z = self.neural_network(t).cpu().numpy().reshape(-1, 1)         # Transform in column vector (repr_dim, 1) for matrix operations in LinUCB
        return z

    # Method to select an arm given a context x, return an int representing the chosen arm
    def select_arm(self, x) -> int:
        x1d = self._flatten_context(x)                      # Flatten the context to 1D vector (n_features,) for the encoder
        z = self._encode_numpy(x1d)                         # Encode context to latent representation z (repr_dim, 1)
        scores = np.zeros(self.n_arms, dtype=np.float64)    
        for arm in range(self.n_arms):                      # For each arm:
            theta = self.A_inv[arm] @ self.b[arm]                       #   - Compute estimated weight vector
            mean_reward = float((theta.T @ z).item())                   #   - Compute mean reward for the arm using the current model (exploitation)
            uncertainty = np.sqrt((z.T @ self.A_inv[arm] @ z).item())   #   - Compute uncertainty for the arm
            scores[arm] = mean_reward + self.alpha * uncertainty        #   Compute final UCB score for the arm
            
        max_score = np.max(scores)
        best_arms = [i for i, v in enumerate(scores) if v == max_score]
        import random
        return int(random.choice(best_arms))  # choose random arm among highest scores

    # Method to update the model parameters after observing a reward for an arm given a context
    def update(self, arm: int, x, reward: float):
        x1d = self._flatten_context(x)                              # Prepare context
        z = self._encode_numpy(x1d)                                 # Encode to latent
        # Sherman-Morrison rank-1 update for A_inv (efficient inverse update)
        num = (self.A_inv[arm] @ z) @ (z.T @ self.A_inv[arm])       
        den = 1.0 + (z.T @ self.A_inv[arm] @ z).item()
        self.A_inv[arm] -= num / den                                # update inverse covariance
        self.b[arm] += float(reward) * z                            # update reward-weighted sum
        # push into replay buffer for encoder training
        self.buffer.append((x1d, int(arm), float(reward)))
        self._update_steps += 1
        # periodically perform a training step on the encoder
        if len(self.buffer) >= self.batch_size and (self._update_steps % self.train_every == 0):
            self._train_step()

    # Method to perform one training step on the neural network using past experiences from the replay buffer
    def _train_step(self):
        batch = random.sample(self.buffer, self.batch_size)                     # Sample a mini-batch of past experiences
        x_batch = np.stack([b[0] for b in batch]).astype(np.float32)            # Shape: (B, n_features)
        arm_batch = np.array([b[1] for b in batch], dtype=np.int64)             # Shape: (B,)
        reward_batch = np.array([b[2] for b in batch], dtype=np.float32)        # Shape: (B,)
        x_t = torch.from_numpy(x_batch).to(self.device)                         # Move the batch to the selected device
        z_t = self.neural_network(x_t)                                          # Shape: (B, repr_dim)
        preds = []
        for i in range(len(batch)):
            arm = int(arm_batch[i])                                             # Arm taken in this experience
            theta = torch.from_numpy(self.A_inv[arm] @ self.b[arm]).float().squeeze(1).to(self.device)  # Estimated weight vector for the arm, shape (repr_dim,)
            zi = z_t[i]                                                         # Latent representation for this sample   
            pred = (zi @ theta).unsqueeze(0)                                    # Predicted reward for this sample
            preds.append(pred)                                                  # Collect predictions for the batch
        pred_tensor = torch.cat(preds).view(-1)                                 # Predicted rewards for the batch
        reward_t = torch.from_numpy(reward_batch).to(self.device)               # Actual rewards for the batch
        loss = self.loss_fn(pred_tensor, reward_t)                              # Compare predicted reward with observed reward
        self.optimizer.zero_grad()                                              # Clear gradients before backward pass
        loss.backward()                                                         # PyTorch calculate gradients and Adam modify the weights 
        torch.nn.utils.clip_grad_norm_(self.neural_network.parameters(), 5.0)   # Clip gradients for stability
        self.optimizer.step()

    # Method to save the full model state to a file.
    # This stores the neural network, the optimizer, and the LinUCB matrices.
    def save(self, path: str):
        payload = {
            "encoder_state": self.neural_network.state_dict(),      # Neural network weights
            "optimizer_state": self.optimizer.state_dict(),         # Optimizer state
            "A_inv": self.A_inv,                                    # Per-arm inverse covariance matrices
            "b": self.b,                                            # Per-arm reward vectors
            "n_arms": self.n_arms,
            "n_features": self.n_features,
            "repr_dim": self.repr_dim,
            "alpha": self.alpha,
        }
        torch.save(payload, path)  # Save everything in one file

    # Method to load a previously saved model state.
    # If some fields are missing, the current values are kept.
    def load(self, path: str):
        try:
            payload = torch.load(path, map_location=self.device, weights_only=False)  # Load trusted checkpoint state
        except TypeError:
            payload = torch.load(path, map_location=self.device)  # Backward compatibility with older PyTorch versions
        if "encoder_state" in payload:
            self.neural_network.load_state_dict(payload["encoder_state"])  # Restore neural network weights
        if "optimizer_state" in payload:
            try:
                self.optimizer.load_state_dict(payload["optimizer_state"])  # Restore optimizer state
            except Exception:
                pass
        self.A_inv = [np.asarray(m, dtype=np.float64) for m in payload.get("A_inv", self.A_inv)]  # Restore A_inv
        self.b = [np.asarray(v, dtype=np.float64) for v in payload.get("b", self.b)]  # Restore b

