import numpy as np

# LinUCB implementation (based on https://arxiv.org/pdf/1003.0146.pdf)
# LinUCB suppose that reward for an arm a given a context x is given by a linear function : μ(a|x) = θ_a^T * x, where: 
#   - θ_a = weight vector for arm a (taille n_features)
#   - x = vector of features for the context (taille n_features)
#   - μ(a|x) = predicted average reward for arm a with context x
# Problem :     We do not know θ_a, but we can estimate it from the data we collect by playing arms and observing reward (context, reward) 
# Solution :    Linear regression with regularization (moindres carrés) to estimate θ_a for each arm a, 
#                   and then use the upper confidence bound to select arms during play (exploration/exploitation tradeoff)
#               For each arm a we maintain:
#                   - A_a = D^T * D + λI (n_features x n_features matrix) : covariance matrix of contexts for arm a with regularization
#                   - b_a = D^T * r (n_features x 1 vector) : cumulative reward-weighted context for arm a
#               Where D is the matrix of contexts for arm a, r is the vector of rewards observed for arm a, and λ is a regularization parameter.
class LinUCB:

    def __init__(self, n_arms: int, n_features: int, alpha: float = 1.5):
        """ The LinUCB algorithm maintains an estimate of the reward function for each arm based on the contexts and rewards observed so far.
        It uses this estimate to select arms in a way that balances exploration and exploitation.

        :param n_arms: Number of arms (actions) in the bandit problem.
        :type n_arms: int
        :param n_features: Dimensionality of the context features used to predict rewards for each arm. 
            Each context is represented as a vector of this size. 
            The context features can include information about the current state of the environment, 
            such as legal moves, time left, move number, etc. 
            The algorithm uses these features to learn the relationship between the context and the rewards for each arm, 
            allowing it to make informed decisions about which arm to select based on the current context.
        :type n_features: int
        :param alpha: Exploration parameter. The higher the more it will explore. >1 for more exploration, <1 for more exploitation, defaults to 1.5
        :type alpha: float, optional
        """
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        # A_a^-1 for each arm a, initialized to identity matrix (regularization)
        self.A_inv = [
            np.identity(n_features) # No observations yet, so A_a = Identity matrix (λI) and A_a^-1 = Identity matrix
            for _ in range(n_arms)  # Used to compute the upper confidence bound and to update the model after observing rewards
        ]
        # b_a for each arm a, initialized to zero vector                                
        self.b = [
            np.zeros((n_features, 1)) # No rewards observed yet, so b_a = 0 vector
            for _ in range(n_arms) # Used to compute the estimated reward for each arm and to update the model after observing rewards
        ]

    # Method to select an arm given a context x, using the upper confidence bound formula
    def select_arm(self, x: np.ndarray) -> int:
        """ Method to select an arm given a context x, using the upper confidence bound formula.

        :param x: Context vector (n_features x 1) for the current decision. 
            This vector should contain the relevant features that the algorithm will use 
            to predict rewards for each arm and make a selection.
        :type x: np.ndarray
        :return: The index of the selected arm (0 to n_arms-1) based on the upper confidence bound calculation.
        :rtype: int
        """
        p_values = []                                       # List to store the upper confidence bound for each arm
        for arm in range(self.n_arms):                      
            theta = self.A_inv[arm] @ self.b[arm]           # @ matrix multiplication to compute Theta : the estimated weight vector for arm a
                                                            # Theta represent the current estimate of relationship between context features and rewards for arm a based on observed data
            p = (                                           # p is the UCB value for arm a given context x, computed as: 
                theta.T @ x                                 #   - T the estimated reward for arm a given context x based on current model (exploitation) multiplied by x (context features)
                + self.alpha                                #   + exploration parameter alpha
                * np.sqrt(x.T @ self.A_inv[arm] @ x))       #   multiplied by the uncertainty term (exploration) : the standard deviation of the reward estimate for arm a given context x, computed as sqrt(x^T * A_a^-1 * x)
                                                            #   --> Standard deviation is needed to balance exploration and exploitation: 
                                                            #       - if the model is uncertain about the reward for arm a given context x (high variance), it will increase the UCB value for that arm, encouraging exploration; 
                                                            #       - if the model is confident (low variance), it will rely more on the estimated reward (exploitation)
                                                            #   --> We do x.T (row vector) * A_a^-1 (matrix) * x (column vector) to compute the variance
                                                            #   --> Variance is computed using A_a^-1 because it captures the amount of information we have about arm a:
                                                            #       - if we have observed many contexts similar to x for arm a, A_a will have large values in the directions of those contexts, making A_a^-1 small in those directions
                                                            #       - if we have observed few contexts similar to x for arm a, A_a will have small values in the directions of those contexts, making A_a^-1 large in those directions, increasing the uncertainty and encouraging exploration 
            p_values.append(p.item())                       # Append the computed UCB value for arm a to the list of p_values
        
        max_p = max(p_values)
        best_arms = [i for i, v in enumerate(p_values) if v == max_p]
        import random
        return int(random.choice(best_arms)) # Return a random choice among the arms with the highest UCB value to avoid sticking to arm 0 at start

    def update(self, arm: int, x: np.ndarray, reward: float) -> None:
        """ Method to update the model after observing a reward for a selected arm and context.

        :param arm: Arm index (0 to n_arms-1) for which the reward was observed. 
            This indicates which arm's model should be updated based on the new information.
        :type arm: int
        :param x: Context vector (n_features x 1) for the current decision.
            This vector should be the same context that was used when selecting the arm, 
            as it is used to update the model based on the observed reward.
        :type x: np.ndarray
        :param reward: Reward observed for the selected arm and context.
        :type reward: float
        """
        num = (self.A_inv[arm] @ x) @ (x.T @ self.A_inv[arm])       # num is the matrix used to update A_a^-1 after observing a reward for arm a given context x, 
                                                                    # Computed using the Sherman-Morrison formula. For the math, see https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
                                                                    # Intuition : 
                                                                    #   - We want to update A_a^-1 to reflect the new information we have about arm a after observing the reward for context x
                                                                    #   - To do so, we update the inverse of A_a using the Sherman-Morrison formula, which allows us to update the inverse of a matrix after a rank-1 update (which is what happens when we add x * x^T to A_a)
                                                                    #   - The formula is: A_a^-1_new = A_a^-1 - (A_a^-1 * x * x^T * A_a^-1) / (1 + x^T * A_a^-1 * x)
                                                                    # --> Updating the inverse directly is more efficient than recomputing it from scratch after updating A_a, which would require inverting a matrix (O(n^3) operation)
        den = 1.0 + (x.T @ self.A_inv[arm] @ x).item()
        self.A_inv[arm] -= num / den
        self.b[arm] += reward * x                                   # Update b_a by adding the observed reward weighted by the context x for arm a, which is used to update our estimate of the relationship between context features and rewards for arm a
                                                                    # We can multiply the reward by x because b_a is the cumulative reward-weighted context for arm a, so we add the new reward weighted by the context to it; x is a colum vector and reward a scalar