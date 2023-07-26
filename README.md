# Reinforcement-Learning-for-Taxi-v3-Environment

This GitHub repository contains code for implementing and evaluating different reinforcement learning algorithms on the Taxi-v3 environment using OpenAI Gym. The project focuses on two approaches: Q-Learning and Q-Network (Deep Q-Network).

1. Q-learning:
The first part of the project implements the Q-learning algorithm to find an optimal policy for the Taxi problem. Q-learning is a model-free, off-policy RL algorithm that learns an action-value function (Q-table) to determine the best action to take in each state. The algorithm is trained for a specified number of episodes, and the performance is plotted in terms of average total reward (mean ± standard deviation) to observe convergence and the effect of the discount factor (γ = 0.9).

2. Q-Network:
The second part replaces the Q-table with a neural network to approximate the Q-values. Two approaches are studied for training data: sampling the Q-table directly or using the Q-table to drive the taxi and store the experience as input-output pairs for training. The neural network architecture comprises a one-hot encoded input representing the states and outputs for each possible action. The network is trained using Mean Squared Error (MSE) loss.

Comparative Analysis:
The Q-learning and Q-Network approaches are compared in terms of convergence speed and performance. The number of training samples needed for the Q-network to approximate Q-table information is also studied. The final output is an optimal policy that guides the taxi to make efficient decisions in the Taxi environment.

Through this project, we gain insights into the strengths and weaknesses of Q-learning and Q-Networks and explore how they can be leveraged for effective reinforcement learning in the context of the Taxi problem.
