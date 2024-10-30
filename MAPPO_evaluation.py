import numpy as np
from MAPPO import MAPPO
from MAPPO_run import save_eval_data
from MAPPO_utils import obs_list_to_state_vector
from NASH import *
from ProductionLine import Multiproduct


def evaluate_from_checkpoint():
    # Custom environment setup
    env = Multiproduct(ptypes, n, b, B, Tp, ng, MTTR, MTBF, T, Tl, Tu)
    batch_size = 64
    alpha = 3e-4

    # Initialize actor dimensions and actions
    actor_dims = env.get_obs_space().shape[0]
    n_actions = env.get_action_space().shape[0]
    critic_dims = actor_dims  # env is completely observable

    # Initialize MAPPO and memory
    mappo_agents = MAPPO(actor_dims=actor_dims, critic_dims=critic_dims,
                         n_agents=env.ng, n_actions=n_actions, alpha=alpha,
                         gamma=0.95)

    # Load the saved checkpoint
    mappo_agents.load_checkpoint()

    # Evaluation settings
    MAX_EVAL_EPISODES = 1  # Number of evaluation episodes
    EVAL_RUNS_PER_EPISODE = 1  # Number of runs per evaluation episode

    # Evaluation storage
    eval_rewards = []
    eval_std_devs = []

    # EVALUATION PHASE
    for eval_episode in range(MAX_EVAL_EPISODES):
        episode_rewards = []

        for eval_run in range(EVAL_RUNS_PER_EPISODE):  # Run each episode multiple times
            observation = env.reset()
            total_reward = 0

            for step in range(100):
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


if __name__ == '__main__':
    evaluate_from_checkpoint()
