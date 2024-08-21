import numpy as np
import pandas as pd

from logbexp.utils.optimization import fit_online_logistic_estimate, fit_online_logistic_estimate_bar
from logbexp.utils.utils import sigmoid, dsigmoid, weighted_norm

# Define theta_star based on the provided values
theta_star = np.array([7.76800624e-01, -6.21092967e-01, -2.63435110e-02, -1.83285908e-01,
                       -1.86187362e-04, 5.23992949e-04, -3.20098726e-02, -7.12018933e-02, -1.38427526e-01])

class ContextualEcoLog():
    def __init__(self, param_norm_ub, arm_norm_ub, context_dim, action_dim, failure_level):
        self.param_norm_ub = param_norm_ub
        self.arm_norm_ub = arm_norm_ub
        self.context_dim = context_dim
        self.action_dim = action_dim
        self.dim = context_dim + action_dim
        self.failure_level = failure_level

        self.name = 'contextualAdaECOLog'
        self.l2reg = 8

        self.vtilde_matrix = self.l2reg * np.eye(self.dim)
        self.vtilde_matrix_inv = (1 / self.l2reg) * np.eye(self.dim)
        self.theta = np.zeros((self.dim,))
        self.conf_radius = 0
        self.cum_loss = 0
        self.ctr = 1

    def reset(self):
        self.vtilde_matrix = self.l2reg * np.eye(self.dim)
        self.vtilde_matrix_inv = (1 / self.l2reg) * np.eye(self.dim)
        self.theta = np.zeros((self.dim,))
        self.conf_radius = 0
        self.cum_loss = 0
        self.ctr = 1

    def learn(self, context, action, reward):
        # Combine context and action into a single feature vector
        feature_vector = np.concatenate((context, action))

        # compute new estimate theta
        self.theta = np.real_if_close(fit_online_logistic_estimate(
            arm=feature_vector,
            reward=reward,
            current_estimate=self.theta,
            vtilde_matrix=self.vtilde_matrix,
            vtilde_inv_matrix=self.vtilde_matrix_inv,
            constraint_set_radius=self.param_norm_ub,
            diameter=self.param_norm_ub,
            precision=1/self.ctr
        ))

        # compute theta_bar (needed for data-dependent conf. width)
        theta_bar = np.real_if_close(fit_online_logistic_estimate_bar(
            arm=feature_vector,
            current_estimate=self.theta,
            vtilde_matrix=self.vtilde_matrix,
            vtilde_inv_matrix=self.vtilde_matrix_inv,
            constraint_set_radius=self.param_norm_ub,
            diameter=self.param_norm_ub,
            precision=1/self.ctr
        ))

        disc_norm = np.clip(weighted_norm(self.theta-theta_bar, self.vtilde_matrix), 0, np.inf)

        # update matrices
        sensitivity = dsigmoid(np.dot(self.theta, feature_vector))
        self.vtilde_matrix += sensitivity * np.outer(feature_vector, feature_vector)
        self.vtilde_matrix_inv += - sensitivity * np.dot(self.vtilde_matrix_inv,
                                                         np.dot(np.outer(feature_vector, feature_vector), self.vtilde_matrix_inv)) / (
                                          1 + sensitivity * np.dot(feature_vector, np.dot(self.vtilde_matrix_inv, feature_vector)))

        # sensitivity check
        sensitivity_bar = dsigmoid(np.dot(theta_bar, feature_vector))
        if sensitivity_bar / sensitivity > 2:
            msg = f"\033[95m Oops. ContextualECOLog has a problem: the data-dependent condition was not met. This is rare; try increasing the regularization (self.l2reg) \033[95m"
            raise ValueError(msg)

        # update sum of losses
        eps = 1e-8
        coeff_theta = np.clip(sigmoid(np.dot(self.theta, feature_vector)), eps, 1 - eps)
        coeff_bar = np.clip(sigmoid(np.dot(theta_bar, feature_vector)), eps, 1 - eps)

        loss_theta = -reward * np.log(coeff_theta) - (1 - reward) * np.log(1 - coeff_theta)
        loss_theta_bar = -reward * np.log(coeff_bar) - (1 - reward) * np.log(1 - coeff_bar)
        self.cum_loss += 2*(1+self.param_norm_ub)*(loss_theta_bar - loss_theta) - 0.5*disc_norm

        # update ctr
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
        self.update_ucb_bonus()

        # Compute optimistic rewards for all arms
        optimistic_rewards = np.array([self.compute_optimistic_reward(context, arm) for arm in arms])

        # Select the arm with the highest optimistic reward
        best_arm_index = np.argmax(optimistic_rewards)
        best_arm = arms[best_arm_index]

        return best_arm, best_arm_index

    def update_ucb_bonus(self):
        """
        Updates the ucb bonus function
        """
        gamma = np.sqrt(self.l2reg) / 2 + 2 * np.log(
            2 * np.sqrt(1 + self.ctr / (4 * self.l2reg)) / self.failure_level) / np.sqrt(self.l2reg)
        res_square = 2*self.l2reg*self.param_norm_ub**2 + (1+self.param_norm_ub)**2*gamma + self.cum_loss
        self.conf_radius = np.sqrt(res_square)

    def compute_optimistic_reward(self, context, arm):
        """
        Returns prediction + exploration_bonus for the given context-arm pair.
        """
        feature_vector = np.concatenate((context, arm))
        norm = weighted_norm(feature_vector, self.vtilde_matrix_inv)
        pred_reward = sigmoid(np.sum(self.theta * feature_vector))
        bonus = self.conf_radius * norm
        return pred_reward + bonus

# setting up the envs and adding datasets
class LogisticBanditEnv(object):

  def __init__(self, theta):
    self.theta = theta

  def step(self, arm):
    p = sigmoid(np.dot(arm, self.theta))
    print(f"p: {p}")
    return np.random.binomial(1, p)

NUM_ROUNDS = 100
NUM_ARM = 10

# Read datasets using pandas
ads_df = pd.read_csv('adsInfoOutput.csv')
customer_info_df = pd.read_csv('audienceInfoOutput.csv')

# Select the first NUM_ARM rows from the ads dataframe
ARMS = ads_df[['Ad_Min_Age', 'Ad_Max_Age', 'Ad_Female', 'Ad_Male', 'Impressions', 'Clicks']].head(NUM_ARM).values
env = LogisticBanditEnv(theta_star)
alg = ContextualEcoLog(param_norm_ub=1, arm_norm_ub=1, context_dim=customer_info_df.shape[1], action_dim=ARMS.shape[1], failure_level=0.1)

for round in range(NUM_ROUNDS):
    context = customer_info_df.sample(1).values.flatten()

    best_arm, best_arm_index = alg.pull(context, ARMS)
    feature_vector = np.concatenate((context, best_arm))
    reward = env.step(feature_vector)
    alg.learn(context, best_arm, reward)

    print(f"Round {round + 1}/{NUM_ROUNDS}")
    print("Context:", context)
    print("Best Arm (Ad) Index:", best_arm_index)
    print("Best Arm (Ad) Features:", best_arm)
    print("Feature Vector:", feature_vector)
    print("Reward:", reward)
    print("-" * 50)
