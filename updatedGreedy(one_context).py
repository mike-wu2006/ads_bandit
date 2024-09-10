import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from logbexp.utils.utils import sigmoid

# Define theta_star based on the provided values
theta_star = np.array([0.17804974, -0.11028392, -0.1086882, -0.16341375, -0.01371968, -0.14434532, -0.12775663])

class GreedyAlgorithm():
    def __init__(self, context_dim, action_dim):
        self.context_dim = context_dim
        self.action_dim = action_dim
        self.dim = context_dim + action_dim
        self.theta = np.zeros((self.dim,))
        self.ctr = 1

    def reset(self):
        self.theta = np.zeros((self.dim,))
        self.ctr = 1

    def learn(self, context, action, reward):
        # Combine context and action into a single feature vector
        feature_vector = np.concatenate((context, action))

        # Update theta using a simple gradient ascent step
        learning_rate = 1.0 / np.sqrt(self.ctr)
        prediction = sigmoid(np.dot(self.theta, feature_vector))
        self.theta += learning_rate * (reward - prediction) * feature_vector

        # Update counter
        self.ctr += 1

    def pull(self, context, arms):
        """
        Select the best arm given the current context and a vector of available arms.

        Parameters:
        context (np.array): The context vector
        arms (np.array): A 2D array where each row represents an arm/action

        Returns:
        np.array: The selected arm
        int: The index of the selected arm
        """
        # Compute rewards for all arms
        rewards = np.array([sigmoid(np.dot(self.theta, np.concatenate((context, arm)))) for arm in arms])

        # Select the arm with the highest reward
        best_arm_index = np.argmax(rewards)
        best_arm = arms[best_arm_index]

        return best_arm, best_arm_index

# Setting up the environment and adding datasets
class LogisticBanditEnv(object):

    def __init__(self, theta):
        self.theta = theta

    def step(self, arm):
        p = sigmoid(np.dot(arm, self.theta))
        return np.random.binomial(1, p)

NUM_ROUNDS = 2500
NUM_ARM = 27

# Read datasets using pandas
ads_df = pd.read_csv('adsInfoOutput.csv')
ARMS = ads_df[['Ad_Min_Age', 'Ad_Max_Age', 'Ad_Female', 'Ad_Male']].head(NUM_ARM).values

fixed_context = np.array([47, 0, 1])

env = LogisticBanditEnv(theta_star)
alg = GreedyAlgorithm(context_dim=fixed_context.shape[0], action_dim=ARMS.shape[1])

cumulative_regret = np.zeros(NUM_ROUNDS)
instant_regret = np.zeros(NUM_ROUNDS)

for round in range(NUM_ROUNDS):
    context = fixed_context
    best_arm, best_arm_index = alg.pull(context, ARMS)
    feature_vector = np.concatenate((context, best_arm))
    reward = env.step(feature_vector)
    alg.learn(context, best_arm, reward)
    true_rewards = np.array([sigmoid(np.dot(np.concatenate((context, arm)), theta_star)) for arm in ARMS])
    optimal_reward = np.max(true_rewards)
    instant_regret[round] = optimal_reward - sigmoid(np.dot(feature_vector, theta_star))
    cumulative_regret[round] = instant_regret[round] + (cumulative_regret[round - 1] if round > 0 else 0)

    print(f"Round {round + 1}/{NUM_ROUNDS}")
    print("Fixed Context (Customer):", context)
    print("Best Arm (Ad) Index:", best_arm_index)
    print("Best Arm (Ad) Features:", best_arm)
    print("Feature Vector:", feature_vector)
    print("Reward:", reward)
    print("Instantaneous Regret:", instant_regret[round])
    print("Cumulative Regret:", cumulative_regret[round])
    print("-" * 50)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(instant_regret, label='Instant Regret', color='orange')
plt.xlabel('Rounds')
plt.ylabel('Instant Regret')
plt.title('Instant Regret over Time')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(cumulative_regret, label='Cumulative Regret', color='blue')
plt.xlabel('Rounds')
plt.ylabel('Cumulative Regret')
plt.title('Cumulative Regret over Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
