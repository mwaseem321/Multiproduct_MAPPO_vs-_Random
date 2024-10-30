import numpy as np
import torch
import matplotlib.pyplot as plt
from MAPPO import MAPPO
from MAPPO_memory import PPOMemory
from MAPPO_utils import obs_list_to_state_vector
from ProductionLine import Multiproduct
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


def run():
    # Custom environment setup
    env = Multiproduct(ptypes, n, b, B, Tp, ng, MTTR, MTBF, T, Tl, Tu)
    batch_size = 64
    alpha = 3e-4

    # Initialize actor dimensions and actions
    actor_dims = env.get_obs_space().shape[0]
    n_actions = env.get_action_space().shape[0]
    critic_dims = actor_dims # env is completely observable

    # Initialize MAPPO and memory
    mappo_agents = MAPPO(actor_dims=actor_dims, critic_dims=critic_dims,
                         n_agents=env.ng, n_actions=n_actions, alpha=alpha,
                        gamma=0.95)

    memory = PPOMemory(batch_size, env.T, env.ng, critic_dims, actor_dims, n_actions)

    # Set number of training episodes and evaluation settings
    MAX_TRAIN_EPISODES = 2000  # Number of training episodes
    MAX_EVAL_EPISODES = 50  # Number of evaluation episodes
    EVAL_RUNS_PER_EPISODE = 25  # Number of runs per evaluation episode to calculate std dev

    # Training storage
    training_rewards = []

    # TRAINING PHASE
    for episode in range(MAX_TRAIN_EPISODES):
        observation = env.reset()
        total_reward = 0
        # print("Episode: ", episode)
        for step in range(env.T):
            # print("Step: ", step)
            actions, probs = mappo_agents.choose_action(observation)
            # print("actions from algorithm: ", actions)
            # print("Observation fed into step ftn: ", observation)
            observation_, reward, done, info = env.step(actions)
            state = obs_list_to_state_vector(observation)
            state_ = obs_list_to_state_vector(observation_)
            done = (step>=env.T)
            memory.store_memory(observation, state, actions, probs, reward,
                                observation_, state_, [1.0 if not done else 0.0])
            # Update for next timestep
            observation = observation_
            total_reward += reward

            # Learn every episode
            if (step + 1) % 500 == 0: #1000
                # print("Started Learning")
                mappo_agents.learn(memory)
                # print("Learning done")
                memory.clear_memory()

        training_rewards.append(total_reward)
        print(f"Training Episode {episode+1}/{MAX_TRAIN_EPISODES}, Total Reward: {total_reward}")

    # Save the trained model using MAPPO's save_checkpoint
    mappo_agents.save_checkpoint()

    # Evaluation storage
    eval_rewards = []
    eval_std_devs = []

    # EVALUATION PHASE
    for eval_episode in range(MAX_EVAL_EPISODES):
        episode_rewards = []

        for eval_run in range(EVAL_RUNS_PER_EPISODE):  # Run each episode multiple times
            observation = env.reset()
            total_reward = 0

            for step in range(env.T):
                actions, probs = mappo_agents.choose_action(observation)
                observation_, reward, done, info = env.step(actions)

                state = obs_list_to_state_vector(observation)
                save_eval_data(state, actions, reward)  # Save state, action, reward to file

                total_reward += reward
                observation = observation_

            episode_rewards.append(total_reward)
            print(f"Evaluation Run {eval_run+1}/{EVAL_RUNS_PER_EPISODE}, Total Reward: {total_reward}")

        # Calculate mean and std dev for this evaluation episode
        mean_reward = np.mean(episode_rewards)
        std_dev_reward = np.std(episode_rewards)
        eval_rewards.append(mean_reward)
        eval_std_devs.append(std_dev_reward)
        print(f"Evaluation Episode {eval_episode+1}/{MAX_EVAL_EPISODES}, Mean Reward: {mean_reward:.1f}, Std Dev: {std_dev_reward:.1f}")

    # Plot training and evaluation rewards
    plot_results(training_rewards, eval_rewards, eval_std_devs)


if __name__ == '__main__':
    run()
