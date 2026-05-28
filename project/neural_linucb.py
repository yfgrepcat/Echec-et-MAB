import random
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# Method to define the neural network encoder 
class RewardNetwork(nn.Module):
    # Method to initialize the network with one hidden layer by default and ReLU activation
    def __init__(self, input_dim: int, hidden_sizes=(32,), repr_dim: int = 16):
        """Initializes the RewardNetwork with specified architecture.

        :param input_dim: The number of input features (context dimension)
        :type input_dim: int
        :param hidden_sizes: The sizes of the hidden layers, defaults to (32,)
        :type hidden_sizes: tuple, optional
        :param repr_dim: The dimension of the latent representation, defaults to 16
        :type repr_dim: int, optional
        """
        super().__init__()

        # Allow `hidden_sizes` to be provided as an int (single layer) or an iterable
        # Needed for initialization from mab_agent, which provides hidden_sizes as an int
        if isinstance(hidden_sizes, int):
            hidden_sizes = (hidden_sizes,)

        layers = []                 # layers is a list that will contain the layers of the network, built sequentially
        in_dim = input_dim          # Number in of feature, the input dimension of first layer
        for h in (hidden_sizes):    # For each hidden layer, we add a linear layer followed by a ReLU activation
            layers.append(nn.Linear(in_dim, h))     # Linear layer is used to transform the input from in_dim to h dimensions, where h is the number of neurons in this hidden layer
            layers.append(nn.ReLU())                # ReLU activation function is applied after the linear transformation to introduce non-linearity, allowing the network to learn more complex representations
            in_dim = h              # Update in_dim for the next layer, the output size of the current layer
        layers.append(              
            nn.Linear(in_dim, repr_dim)
        )                           # Final layer to output the latent reprensentaiton,
        layers.append(nn.ReLU())
        self.net = nn.Sequential(
            *layers
        )                           # Sequential permit to chain the layer together and simplify forward pass 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network to get the latent representation.

        :param x: Input tensor containing the context features, shape (batch_size, input_dim)
        :type x: torch.Tensor
        :return: Latent representation of the input context, shape (batch_size, repr_dim)
        :rtype: torch.Tensor
        """
        return self.net(x)


# NeuralLinUCB is a contextual bandit algorithm that uses a neural network to learn a latent representation of the context, then applies LinUCB in that learned space.
# The encoder transforms the raw chess features into a fixed-size embedding z, which is not necessarily smaller than the input dimension: it is a learned representation, not a guaranteed compression.
# LinUCB then operates on z instead of directly on the raw features, so the bandit stays linear in the learned space while the overall model is non-linear in the original input.
# The goal is to make the context easier to exploit for the bandit by learning useful combinations of the 7 input features, which can improve exploration/exploitation decisions.
# Implementation of this neural LinUCB is inspired by the paper "Neural Linear Bandits: Overcoming Catastrophic Forgetting through Likelihood Matching" (https://arxiv.org/pdf/1901.08612) and adapted to our chess context and constraints.
# Each hidden neuron computes a weighted sum of the input plus a bias, u = w·x + b, then applies ReLU element-wise: a = max(0, u). The output of all neurons forms the layer representation.
# The neural network is trained periodically (every train_every steps) from a replay buffer storing past contexts, actions, and rewards, so the encoder keeps improving over time.
# The latent size is a design choice: it is fixed by the architecture and does not come directly from the data.
# This version is inspired by Neural Linear Bandits, but kept intentionally small.
class NeuralLinUCB:
    # The encoder takes 7 input features and outputs a latent vector of size representation_dim.
    def __init__(
        self,
        n_arms: int,                    # Number of arms (actions)
        n_features: int,                # Input features
        alpha: float = 1.5,             # Exploration parameter (just like in basic LinUCB)
        hidden_sizes: int = 10,         # One hidden layer: 10 neurons, simple and enough for the current 7 features
        representation_dim: int = 16,   # Output size of the encoder; this is the latent context size used by LinUCB
        lr: float = 1e-3,               # Learning rate for the encoder network
        ridge_lambda: float = 1.0,      # Regularization (ridge) for the linear part; affects init scale of A_inv
        batch_size: int = 32,           # Number of past training examples sampled at once from the replay buffer
        train_every: int = 5,           # Perform a training step every N updates
        replay_size: int = 10000,       # Maximum size of the replay buffer
        seed: int = 42,                 # Random seed for reproducibility
        device: str = "auto",           # Device hint for torch: 'auto'|'cpu'|'cuda'|'mps'
        force_cpu: bool = False,        # If True, force CPU even when GPU is available
    ):
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        self.repr_dim = representation_dim
        self.batch_size = batch_size
        self.train_every = max(1, int(train_every))
        self.ridge_lambda = ridge_lambda
        self._rng = np.random.default_rng(
            seed
        )  # Seeds permit to have reproductible results
        random.seed(
            seed
        )  # Seed for python random, used in select_arm to break ties randomly
        torch.manual_seed(
            seed
        )  # Initialization of weights in the encoder, deterministic with the same seed
        self.device = self._resolve_device(device, force_cpu)  # (CPU/GPU)
        self.neural_network = RewardNetwork(
            n_features, hidden_sizes, representation_dim
        ).to(
            self.device
        )  # Encoder is the neural network in charge of creating the latent representation
        self.optimizer = torch.optim.Adam(
            self.neural_network.parameters(), lr=lr
        )  # Adam optimizer, is here to adapt weights for each encoder neuron during training
        self.loss_fn = nn.MSELoss()  # Prediction loss, using MSE (mean squared error) between predicted reward and observed reward during training of the encoder
        init_scale = 1.0 / float(ridge_lambda)
        self.A_inv = [
            np.identity(representation_dim, dtype=np.float64) * init_scale
            for _ in range(n_arms)
        ]  # Initial A_a^-1 for each arm a (identity matrix)
        self.b = [
            np.zeros((representation_dim, 1), dtype=np.float64) for _ in range(n_arms)
        ]  # Initial b_a for each arm a (zero vector) : vector reward
        self.buffer = deque(
            maxlen=int(replay_size)
        )  # Replay memory of played actions for training
        self._update_steps = 0  # Counter to schedule training, every train_every steps

    @staticmethod
    def _resolve_device(device: str, force_cpu: bool):
        """ This method checks the provided device string and the availability 
        of CUDA to determine whether to use CPU or GPU for computations.
        :param device: A string indicating the desired device ('auto', 'cpu', 'cuda', 'mps')
        :type device: str
        :param force_cpu: A boolean flag that, if True, forces the use of CPU even if a GPU is available. Defaults to False.
        :type force_cpu: bool
        :return: A torch.device object representing the selected device for computations.
        :rtype: torch.device
        """
        if force_cpu:
            return torch.device("cpu")  # Force CPU when requested
        if device == "cpu":
            return torch.device("cpu")
        if device == "cuda":
            return torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        # Try cuda if available and no cpu or cuda requested
        if torch.cuda.is_available():
            return torch.device("cuda")
        # Otherwise CPU
        return torch.device("cpu")

    def _flatten_context(self, x) -> np.ndarray:
        """ Flattens the input context into a 1D NumPy array of type float32,
        ensuring it has the correct shape for the encoder.
        :param x: The input context features.
        :type x: np.ndarray
        :raises ValueError: If the input context has an incorrect shape.
        :return: The flattened context as a 1D NumPy array.
        :rtype: np.ndarray
        """
        arr = np.asarray(x, dtype=np.float32).reshape(
            -1
        )  # Convert into NumPy array, force float32,flatten to 1D
        if (
            arr.shape[0] != self.n_features
        ):  # Vector length need to be exactly n_features
            raise ValueError(
                f"Context size mismatch: expected {self.n_features}, got {arr.shape[0]}"
            )
        return arr

    def _encode_numpy(self, x1d: np.ndarray) -> np.ndarray:
        """ Encodes the flattened context vector into a latent 
        representation using the neural network encoder.
        
        :param x1d: A 1D NumPy array containing the context features, with shape (n_features,) and dtype float32. This is the output of the _flatten_context method, which prepares the raw context for encoding.
        :type x1d: np.ndarray
        :return: A 1D torch tensor containing the latent representation of the input context, with shape (repr_dim,).
        :rtype: np.ndarray
        """
        t = (
            torch.from_numpy(x1d).float().unsqueeze(0).to(self.device)
        )  # PyToroch tensor shared in memory
        with torch.no_grad():  # No gradient needed for encoding during action selection
            z = (
                self.neural_network(t).cpu().numpy().reshape(-1, 1)
            )  # Transform in column vector (repr_dim, 1) for matrix operations in LinUCB
        return z

    def select_arm(self, x: np.ndarray) -> int:
        """ Selects an arm based on the current context x using the UCB strategy.
        The method first encodes the context into a latent representation using the 
        neural network, then computes the UCB score for each arm based on the estimated 
        reward and uncertainty, and finally selects the arm with the highest UCB score.

        :param x: The context features associated with the action taken, which will be used to update the model. This should be in the same format as the input to select_arm (e.g., a list or array of features).
        :type x: np.ndarray
        :return: The index of the selected arm.
        :rtype: int
        """
        x1d = self._flatten_context(
            x
        )  # Flatten the context to 1D vector (n_features,) for the encoder
        z = self._encode_numpy(
            x1d
        )  # Encode context to latent representation z (repr_dim, 1)
        scores = np.zeros(self.n_arms, dtype=np.float64)
        for arm in range(self.n_arms):  # For each arm:
            theta = self.A_inv[arm] @ self.b[arm]  # Compute estimated weight vector
            mean_reward = float(
                (theta.T @ z).item()
            )  # Compute mean reward for the arm using the current model (exploitation)
            uncertainty = np.sqrt(
                (z.T @ self.A_inv[arm] @ z).item()
            )  # Compute uncertainty for the arm
            scores[arm] = (
                mean_reward + self.alpha * uncertainty
            )  # Compute final UCB score for the arm

        max_score = np.max(scores)
        best_arms = [i for i, v in enumerate(scores) if v == max_score]
        import random

        return int(random.choice(best_arms)) # choose random arm among highest scores

    def update(self, arm: int, x: np.ndarray, reward: float):
        """Updates the model parameters based on the observed reward for 
        a given arm and context. This method performs a rank-1 update of 
        the inverse covariance matrix A_inv for the selected arm using the 
        Sherman-Morrison formula, updates the reward vector b for that arm, 
        and stores the experience in the replay buffer for future training of the encoder.

        :param arm: The index of the arm that was selected and for which the reward was observed
        :type arm: int
        :param x: The context features associated with the action taken, which will be used to update the model. This should be in the same format as the input to select_arm (e.g., a list or array of features).
        :type x: list or np.ndarray
        :param reward: The reward received for taking the action corresponding to the selected arm in the given context. This should be a numerical value (float) that represents the outcome of the action.
        :type reward: float
        """
        x1d = self._flatten_context(x)  # Prepare context
        z = self._encode_numpy(x1d)  # Encode to latent
        # Sherman-Morrison rank-1 update for A_inv (efficient inverse update)
        num = (self.A_inv[arm] @ z) @ (z.T @ self.A_inv[arm])
        den = 1.0 + (z.T @ self.A_inv[arm] @ z).item()
        self.A_inv[arm] -= num / den  # update inverse covariance
        self.b[arm] += float(reward) * z  # update reward-weighted sum
        # push into replay buffer for encoder training
        self.buffer.append((x1d, int(arm), float(reward)))
        self._update_steps += 1
        # periodically perform a training step on the encoder
        if len(self.buffer) >= self.batch_size and (
            self._update_steps % self.train_every == 0
        ):
            self._train_step()

    def _train_step(self):
        """
        Performs a training step on the encoder network using a mini-batch of past experiences 
        from the replay buffer. This method samples a batch of (context, arm, reward) tuples, 
        computes the predicted rewards using the current encoder and LinUCB parameters, and updates 
        the encoder weights to minimize the prediction error.
        """
        batch = random.sample(
            self.buffer, self.batch_size
        )  # Sample a mini-batch of past experiences
        x_batch = np.stack([b[0] for b in batch]).astype(
            np.float32
        )  # Shape: (B, n_features)
        arm_batch = np.array([b[1] for b in batch], dtype=np.int64)  # Shape: (B,)
        reward_batch = np.array([b[2] for b in batch], dtype=np.float32)  # Shape: (B,)
        x_t = torch.from_numpy(x_batch).to(
            self.device
        )  # Move the batch to the selected device
        z_t = self.neural_network(x_t)  # Shape: (B, repr_dim)
        preds = []
        for i in range(len(batch)):
            arm = int(arm_batch[i])  # Arm taken in this experience
            theta = (
                torch.from_numpy(self.A_inv[arm] @ self.b[arm])
                .float()
                .squeeze(1)
                .to(self.device)
            )  # Estimated weight vector for the arm, shape (repr_dim,)
            zi = z_t[i]  # Latent representation for this sample
            pred = (zi @ theta).unsqueeze(0)  # Predicted reward for this sample
            preds.append(pred)  # Collect predictions for the batch
        pred_tensor = torch.cat(preds).view(-1)  # Predicted rewards for the batch
        reward_t = torch.from_numpy(reward_batch).to(
            self.device
        )  # Actual rewards for the batch
        loss = self.loss_fn(
            pred_tensor, reward_t
        )  # Compare predicted reward with observed reward
        self.optimizer.zero_grad()  # Clear gradients before backward pass
        loss.backward()  # PyTorch calculate gradients and Adam modify the weights
        torch.nn.utils.clip_grad_norm_(
            self.neural_network.parameters(), 5.0
        )  # Clip gradients for stability
        self.optimizer.step()

    def save(self, path: str):
        """ Saves the model state to a file, including the neural network weights, optimizer state, and LinUCB parameters.

        :param path: Path to save the model file
        :type path: str
        """
        payload = {
            "encoder_state": self.neural_network.state_dict(),  # Neural network weights
            "optimizer_state": self.optimizer.state_dict(),  # Optimizer state
            "A_inv": self.A_inv,  # Per-arm inverse covariance matrices
            "b": self.b,  # Per-arm reward vectors
            "n_arms": self.n_arms,
            "n_features": self.n_features,
            "repr_dim": self.repr_dim,
            "alpha": self.alpha,
        }
        torch.save(payload, path) # Save everything in one file

    def load(self, path: str):
        """ Loads the model state from a file, restoring the neural network, optimizer, and LinUCB parameters.

        :param path: Path to the saved model file
        :type path: str
        """
        try:
            payload = torch.load(
                path, map_location=self.device, weights_only=False
            )  # Load trusted checkpoint state
        except TypeError:
            payload = torch.load(
                path, map_location=self.device
            )  # Backward compatibility with older PyTorch versions
        if "encoder_state" in payload:
            self.neural_network.load_state_dict(
                payload["encoder_state"]
            )  # Restore neural network weights
        if "optimizer_state" in payload:
            try:
                self.optimizer.load_state_dict(
                    payload["optimizer_state"]
                )  # Restore optimizer state
            except Exception:
                pass
        self.A_inv = [
            np.asarray(m, dtype=np.float64) for m in payload.get("A_inv", self.A_inv)
        ]  # Restore A_inv
        self.b = [
            np.asarray(v, dtype=np.float64) for v in payload.get("b", self.b)
        ]  # Restore b
