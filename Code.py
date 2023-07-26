import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

## 1)

# Create environment
env = gym.make("Taxi-v3")
env.reset()

def eval_policy(env_, pi_, gamma_):
    env_.reset()

    s_t = env.reset()
    v_pi = 0
    for t in range(666):
        a_t = pi_[s_t]
        s_t, r_t, done, info = env_.step(a_t) 
        v_pi += r_t
        if done:
            break
    env.close()
    return v_pi


# Initialize Q-table
action_size = env.action_space.n
print("Action size: ", action_size)

state_size = env.observation_space.n
print("State size: ", state_size)

q_table = np.zeros([env.observation_space.n, env.action_space.n])
print(q_table)

# Set hyperparameters
alpha = 1
gamma = 0.9
epsilon = 0.8
num_episodes = 600

# Initialize lists to store rewards and evaluation results
rewards = []
eval_results = []

# Q-learning algorithm
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Choose action using epsilon-greedy policy
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state, :])

        # Take action and observe next state and reward
        next_state, reward, done, info = env.step(action)

        # Update Q-table
        q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action])

        # Update state and total reward
        state = next_state
        total_reward += reward

    # Store total reward for this episode
    rewards.append(total_reward)

    # Evaluate policy every 10 episodes
    if episode % 10 == 0:
        eval_rewards = []
        for i in range(100):
            eval_rewards.append(eval_policy(env, np.argmax(q_table, axis=1), gamma))


        # Calculate mean and standard deviation of evaluation rewards
        eval_mean = np.mean(eval_rewards)
        eval_std = np.std(eval_rewards)

        # Append results to evaluation results list
        eval_results.append([episode, eval_mean, eval_std])

# Plot performance
eval_results = np.array(eval_results)
plt.errorbar(eval_results[:, 0], eval_results[:, 1], yerr=eval_results[:, 2], capsize=2)
plt.xlabel('Episodes')
plt.ylabel('Average Total Reward')
plt.show()

# Print optimal policy
optimal_policy = np.argmax(q_table, axis=1)
print(optimal_policy)




## 2) You sample the Q-table

# Create environment
env = gym.make("Taxi-v3")
env.reset()

# One-hot encoding for states
def one_hot_state(state):
    state_enc = np.zeros(env.observation_space.n)
    state_enc[state] = 1.0
    return state_enc

def eval_policy(env_, pi_, gamma_):
    env_.reset()

    s_t = env.reset()
    v_pi = 0
    for t in range(666):
        a_t = pi_[np.argmax(one_hot_state(s_t))]
        s_t, r_t, done, info = env_.step(a_t) 
        v_pi += r_t
        if done:
            break
    env.close()
    return v_pi


# Initialize Q-table
action_size = env.action_space.n
print("Action size: ", action_size)

state_size = env.observation_space.n
print("State size: ", state_size)

q_table = np.zeros([env.observation_space.n, env.action_space.n])
print(q_table)

# Set hyperparameters
alpha = 1
gamma = 0.9
epsilon = 0.8
num_episodes = 3000

# Initialize lists to store rewards and evaluation results
rewards = []
eval_results = []

# Q-learning algorithm
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Choose action using epsilon-greedy policy
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            # Select action randomly from among the best options for sampling
            best_actions = np.flatnonzero(q_table[state, :] == np.max(q_table[state, :]))
            action = np.random.choice(best_actions)
        
        # Take action and observe next state and reward
        next_state, reward, done, info = env.step(action)

        # Update Q-table
        q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action])
                
        # Update state and total reward
        state = next_state
        total_reward += reward

    # Store total reward for this episode
    rewards.append(total_reward)

    # Evaluate policy every 10 episodes
    if episode % 10 == 0:
        eval_rewards = []
        for i in range(100):
            eval_rewards.append(eval_policy(env, np.argmax(q_table, axis=1), gamma))

        # Calculate mean and standard deviation of evaluation rewards
        eval_mean = np.mean(eval_rewards)
        eval_std = np.std(eval_rewards)

        # Append results to evaluation results list
        eval_results.append([episode, eval_mean, eval_std])

# Convert the sampled Q-table to training data for neural network 
train_states = np.array([one_hot_state(i) for i in range(state_size)])
train_targets = q_table

# Define neural network
model = Sequential()
model.add(Dense(32, input_dim=state_size, activation='relu'))
model.add(Dense(32, activation='relu')) 
model.add(Dense(action_size, activation='linear'))

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=tf.keras.losses.MeanSquaredError())

# Train model on Q-table data
history = model.fit(train_states, train_targets, epochs=1000)

# Get predictions for optimal policy
optimal_policy = np.argmax(model.predict(train_states), axis=1)
print(optimal_policy)

#Evaluate performance of optimal policy
rewards = []
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = optimal_policy[state]
        next_state, reward, done, info = env.step(action)
        state = next_state
        total_reward += reward

    rewards.append(total_reward)

    # Print progress
    if episode % 10 == 0:
        print(f"Episode {episode}: mean reward = {np.mean(rewards[-100:])}")

# Calculate mean and standard deviation of rewards
mean_rewards = [np.mean(rewards[i:i+100]) for i in range(0, num_episodes, 100)]
std_rewards = [np.std(rewards[i:i+100]) for i in range(0, num_episodes, 100)]

#Plot performance
plt.plot(range(num_episodes), rewards)
plt.xlabel('Episodes')
plt.ylabel('Total Rewards')
plt.title('Performance of Optimal Policy')
plt.show()

# Plot performance with error bars
plt.errorbar(range(0, num_episodes, 100), mean_rewards, yerr=std_rewards, fmt='-o')
plt.xlabel('Episodes')
plt.ylabel('Total Rewards')
plt.title('Performance of Optimal Policy')
plt.show()
