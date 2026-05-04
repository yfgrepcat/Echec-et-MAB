# Chessomatic
An intelligent chess AI selector leveraging the Multi-Armed Bandits algorithm in order to choose the appropriate chess AI based on multiple parameters in a time constrained game of chess.

## MAB Algorithms

Implemented:
- Epsilon-Greedy algorithm (basic exploration/exploitation MAB)
- Thompson Sampling algorithm (basic exploration/exploitation MAB)
- LinUCB algorithm for contextual bandits (deterministic contextual MAB with linear reward model)

To be implemented:
- NeuralBandits algorithm (contextual no guarantees/blackbox MAB with a neural network reward model)
- Contextual Thompson Sampling algorithm (Maybe?)