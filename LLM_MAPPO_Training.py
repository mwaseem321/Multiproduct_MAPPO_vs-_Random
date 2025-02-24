import torch
import matplotlib.pyplot as plt
from MAPPO import MAPPO
from MAPPO_memory import PPOMemory
from MAPPO_utils import obs_list_to_state_vector
from ProductionLine import Multiproduct
import random
from NASH import *


def save_eval_data(state, action, reward, file_name="eval_data.txt"):
    with open(file_name, "a") as file:
        file.write(f"State: {state}, Action: {action}, Reward: {reward}\n")


def plot_results(training_rewards, eval_rewards, eval_std_devs):
    plt.figure(figsize=(12, 6))
    plt.plot(training_rewards, label='Training Reward')
    plt.errorbar(range(len(eval_rewards)), eval_rewards, yerr=eval_std_devs, label='Evaluation Reward (with Std Dev)', fmt='-o')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.legend()
    plt.title('Training and Evaluation Rewards Over Episodes (with Std Dev)')
    plt.show()


import random
import numpy as np
import pickle  # Or any other format for loading your existing data

# Function to load the existing data from a file
def load_existing_data(filename="existing_data.pkl"):
    with open(filename, "rb") as f:
        data = pickle.load(f)  # Assuming data is stored in a pickle file
    return data

# Function to get action from the existing data
def get_action_from_data(state, episode_state_action_maps):
    # Iterate through the episode state-action maps
    for episode_map in episode_state_action_maps:
        if state in episode_map:
            return episode_map[state]  # Return the corresponding action for the state
    return None  # If not found, return None or a default action

#####################################################################
# def run():
#     # Load existing data (already formatted as discussed)
#     episode_state_action_maps = load_existing_data("episodes_data.pkl")
#
#     # Custom environment setup
#     env = Multiproduct(ptypes, n, b, B, Tp, ng, MTTR, MTBF, T, Tl, Tu)
#     batch_size = 64
#     alpha = 3e-4
#
#     # Initialize actor dimensions and actions
#     actor_dims = env.get_obs_space().shape[0]
#     n_actions = env.get_action_space().shape[0]
#     critic_dims = actor_dims  # env is completely observable
#
#     # Initialize MAPPO and memory
#     mappo_agents = MAPPO(actor_dims=actor_dims, critic_dims=critic_dims,
#                          n_agents=env.ng, n_actions=n_actions, alpha=alpha,
#                          gamma=0.95)
#     memory = PPOMemory(batch_size, env.T, env.ng, critic_dims, actor_dims, n_actions)
#
#     # Set number of training episodes and evaluation settings
#     MAX_TRAIN_EPISODES = 2000
#     MAX_EVAL_EPISODES = 50
#     EVAL_RUNS_PER_EPISODE = 25
#
#     # Exploration-exploitation epsilon
#     epsilon = 0.1  # Set the epsilon value
#
#     # Training storage
#     training_rewards = []
#
#     # TRAINING PHASE
#     for episode in range(MAX_TRAIN_EPISODES):
#         observation = env.reset()
#         total_reward = 0
#
#         for step in range(env.T):
#             # Exploration vs Exploitation decision (epsilon-greedy)
#             if random.random() < epsilon:
#                 # Exploit the existing data
#                 action = get_action_from_data(tuple(observation), episode_state_action_maps)
#                 if action is None:
#                     # If action is not found in the data, use MAPPO
#                     actions, probs = mappo_agents.choose_action(observation)
#                 else:
#                     actions = action  # Use the action from data
#             else:
#                 # Explore with MAPPO
#                 actions, probs = mappo_agents.choose_action(observation)
#
#             observation_, reward, done, info = env.step(actions)
#             state = obs_list_to_state_vector(observation)
#             state_ = obs_list_to_state_vector(observation_)
#
#             done = (step >= env.T)
#             memory.store_memory(observation, state, actions, probs, reward, observation_, state_, [1.0 if not done else 0.0])
#
#             # Update for next timestep
#             observation = observation_
#             total_reward += reward
#
#             # Learn every episode
#             if (step + 1) % 500 == 0:
#                 mappo_agents.learn(memory)
#                 memory.clear_memory()
#
#         training_rewards.append(total_reward)
#         print(f"Training Episode {episode+1}/{MAX_TRAIN_EPISODES}, Total Reward: {total_reward}")
#
#     # Save the trained model using MAPPO's save_checkpoint
#     mappo_agents.save_checkpoint()
#
#     # EVALUATION PHASE
#     eval_rewards = []
#     eval_std_devs = []
#
#     for eval_episode in range(MAX_EVAL_EPISODES):
#         episode_rewards = []
#
#         for eval_run in range(EVAL_RUNS_PER_EPISODE):
#             observation = env.reset()
#             total_reward = 0
#
#             for step in range(env.T):
#                 actions, probs = mappo_agents.choose_action(observation)
#                 observation_, reward, done, info = env.step(actions)
#
#                 state = obs_list_to_state_vector(observation)
#                 save_eval_data(state, actions, reward)
#
#                 total_reward += reward
#                 observation = observation_
#
#             episode_rewards.append(total_reward)
#             print(f"Evaluation Run {eval_run+1}/{EVAL_RUNS_PER_EPISODE}, Total Reward: {total_reward}")
#
#         mean_reward = np.mean(episode_rewards)
#         std_dev_reward = np.std(episode_rewards)
#         eval_rewards.append(mean_reward)
#         eval_std_devs.append(std_dev_reward)
#         print(f"Evaluation Episode {eval_episode+1}/{MAX_EVAL_EPISODES}, Mean Reward: {mean_reward:.1f}, Std Dev: {std_dev_reward:.1f}")
#
#     plot_results(training_rewards, eval_rewards, eval_std_devs)
#
#
# if __name__ == '__main__':
#     run()


#################### Episode based ##############################

import random
import numpy as np
import pickle  # Or any other format for loading your existing data

# Function to load the existing data from a file
def load_existing_data(filename="episodes_data.pkl"):
    with open(filename, "rb") as f:
        data = pickle.load(f)  # Assuming data is stored in a pickle file
    # print("Data from pkl: ", len(data[0]))
    return data

# Function to get action from the existing data
def get_action_from_data(state, episode_state_action_maps):
    # Iterate through the episode state-action maps (list of dictionaries)
    for episode_map in episode_state_action_maps:
        # Check if the state exists in the episode's state-action map
        if state in episode_map:
            return episode_map[state]  # Return the corresponding action for the state
    return None  # If not found, return None or a default action

action_lookup_list = [(0, 0), (0, 1), (0, 2),(0, 3), (1, 0), (1, 1),(1, 2),(1, 3), (2, 0), (2, 1),(2, 2),(2, 3)]
def map_action_to_index(action, action_lookup_list):
    action_indices = []
    for act in action:
        # Find the index of the action tuple in the lookup list
        if act in action_lookup_list:
            action_indices.append(action_lookup_list.index(act))
    return action_indices

def run():
    # Load existing data (already formatted as discussed)
    episode_state_action_maps = load_existing_data("episodes_data.pkl")

    # Custom environment setup
    env = Multiproduct(ptypes, n, b, B, Tp, ng, MTTR, MTBF, T, Tl, Tu)
    batch_size = 64
    alpha = 0.001 #3e-4

    # Initialize actor dimensions and actions
    actor_dims = env.get_obs_space().shape[0]
    n_actions = env.get_action_space().shape[0]
    critic_dims = actor_dims  # env is completely observable

    # Initialize MAPPO and memory
    mappo_agents = MAPPO(actor_dims=actor_dims, critic_dims=critic_dims,
                         n_agents=env.ng, n_actions=n_actions, alpha=alpha,
                         gamma=0.95)
    memory = PPOMemory(batch_size, env.T, env.ng, critic_dims, actor_dims, n_actions)

    # Set number of training episodes and evaluation settings
    MAX_TRAIN_EPISODES = 10000
    MAX_EVAL_EPISODES = 50
    EVAL_RUNS_PER_EPISODE = 25

    # Exploration-exploitation epsilon
    zeta = 0.8
    min_zeta = 0
    zeta_decay = 0.995
    epsilon = 0.8  # Start with full exploration
    min_epsilon = 0.1  # Minimum epsilon value to ensure some exploration remains
    decay_rate = 0.995  # Rate at which epsilon decays after each episode

    # Training storage
    training_rewards = []

    # TRAINING PHASE
    for episode in range(MAX_TRAIN_EPISODES):
        observation = env.reset()
        total_reward = 0

        # Decide whether to train based on MAPPO or dataset for the entire episode
        if random.random() < zeta:
            # Use dataset for the whole episode
            is_using_dataset = True
        else:
            # Use MAPPO for the whole episode
            is_using_dataset = False

        # Process the episode based on the selected training method
        if is_using_dataset:
            # print("Dataset is being used")
            # Randomly select an episode from the dataset
            selected_episode = random.choice(episode_state_action_maps)
            # Use this entire episode for training
            previous_reward = 0  # Initialize the previous reward as 0 for the first step
            for step in range(len(selected_episode)):
                state, action, reward, new_state, done = selected_episode[step]
                # Calculate step-wise reward (difference between current and previous reward)
                step_reward = reward - previous_reward
                # Update previous reward for the next step
                previous_reward = reward
                action_indices = map_action_to_index(action, action_lookup_list)

                # Create a dictionary where the agent indices map to the action indices
                action_dict = {i: action_indices[i] for i in range(len(action_indices))}

                # Create probs as a dictionary with 1.0 for each agent (fixed probs for dataset training)
                probs = {i: 1.0 for i in range(env.ng)}
                total_reward += step_reward
                # print("state: ", state)
                # print("action: ", action)
                # print("reward: ", step_reward)
                # print("new_state: ", new_state)
                # Store memory with fixed probabilities (no exploration)
                memory.store_memory(state, state, action_dict, probs, step_reward, new_state, new_state,
                                    [1.0 if not done else 0.0])

            # After the episode is processed, we can learn
            mappo_agents.learn(memory)
            memory.clear_memory()
        else:
            # Use MAPPO for the entire episode
            for step in range(env.T):
                # Exploration-exploitation: Decide whether to choose a random action or use policy
                if random.random() < epsilon:
                    # Explore: Choose random action
                    actions = {i: np.random.randint(0, n_actions) for i in range(env.ng)}  # Random action for each agent
                    # print("random actions in mappo: ", actions)
                    probs = {i: 1.0 for i in range(env.ng)}  # No exploration for prob, just fixed at 1
                else:
                    # Exploit: Use MAPPO to choose action
                    actions, probs = mappo_agents.choose_action(observation)

                # Take the chosen action in the environment
                observation_, reward, done, info = env.step(actions)

                # Convert to state vectors for memory storage
                state = obs_list_to_state_vector(observation)
                state_ = obs_list_to_state_vector(observation_)

                # Check if the episode has ended
                done = (step >= env.T)

                # Store the memory with fixed probabilities (no exploration)
                memory.store_memory(observation, state, actions, probs, reward, observation_, state_,
                                    [1.0 if not done else 0.0])

                # Update for next timestep
                observation = observation_
                total_reward += reward

                # Learn every episode (you can adjust this condition to train more frequently)
                if (step + 1) % 240 == 0:
                    mappo_agents.learn(memory)
                    memory.clear_memory()

        # Decay epsilon after each episode
        epsilon = max(min_epsilon, epsilon * decay_rate)
        zeta = max(min_zeta, zeta * zeta_decay)

        # Append the total reward for the episode
        training_rewards.append(total_reward)
        print(f"Training Episode {episode + 1}/{MAX_TRAIN_EPISODES}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}, Zeta: {zeta:.4f}")

    # Save the trained model using MAPPO's save_checkpoint
    mappo_agents.save_checkpoint()

    # EVALUATION PHASE
    eval_rewards = []
    eval_std_devs = []

    for eval_episode in range(MAX_EVAL_EPISODES):
        episode_rewards = []

        for eval_run in range(EVAL_RUNS_PER_EPISODE):
            observation = env.reset()
            total_reward = 0

            for step in range(env.T):
                actions, probs = mappo_agents.choose_action(observation)
                observation_, reward, done, info = env.step(actions)

                state = obs_list_to_state_vector(observation)
                save_eval_data(state, actions, reward)

                total_reward += reward
                observation = observation_

            episode_rewards.append(total_reward)
            print(f"Evaluation Run {eval_run + 1}/{EVAL_RUNS_PER_EPISODE}, Total Reward: {total_reward}")

        mean_reward = np.mean(episode_rewards)
        std_dev_reward = np.std(episode_rewards)
        eval_rewards.append(mean_reward)
        eval_std_devs.append(std_dev_reward)
        print(
            f"Evaluation Episode {eval_episode + 1}/{MAX_EVAL_EPISODES}, Mean Reward: {mean_reward:.1f}, Std Dev: {std_dev_reward:.1f}")

    plot_results(training_rewards, eval_rewards, eval_std_devs)


if __name__ == '__main__':
    run()


