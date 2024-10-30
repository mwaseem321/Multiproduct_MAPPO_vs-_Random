import numpy as np
import torch as T
from MAPPO_networks import DiscreteActorNetwork, CriticNetwork


class Agent:
    def __init__(self, actor_dims, critic_dims,
                 n_actions, agent_idx,
                 gamma=0.99, alpha=3e-4,
                 gae_lambda=0.95, policy_clip=0.2,
                 n_epochs=10,
                 chkpt_dir=None):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coefficient = 1e-3
        self.agent_idx = agent_idx


        self.actor = DiscreteActorNetwork(n_actions, actor_dims, alpha,
                                            chkpt_dir=chkpt_dir)
        self.critic = CriticNetwork(critic_dims, alpha,
                                              chkpt_dir=chkpt_dir)
        self.n_actions = n_actions

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        # print("observation in the choose action: ", observation)
        with T.no_grad():
            state = T.tensor(observation, dtype=T.float, device=self.actor.device)

            # Get the action distribution from the actor network
            dist = self.actor(state)

            # Sample a single action for this agent (since this is one agent)
            action = dist.sample()

            # Log probability of the chosen action
            probs = dist.log_prob(action)
        # print("action.cpu().numpy()",action.cpu().numpy())
        # Return the selected action and its log probability
        return action.cpu().numpy(), probs.cpu().numpy()

    def calc_adv_and_returns(self, memories):
        states, new_states, r, dones = memories
        with T.no_grad():
            values = self.critic(states).squeeze()
            values_ = self.critic(new_states).squeeze()
            # print("values.shape: ", values.shape)
            # print("values_.shape: ", values_.shape)
            # print("r.shape before: ", r.shape)
            r = r.expand(-1, values.size(1))
            # print("r.shape: ", r.shape)
            deltas = r + self.gamma * values_ - values
            deltas = deltas.cpu().numpy()
            adv = [0]
            for step in reversed(range(deltas.shape[0])):
                advantage = deltas[step] +\
                    self.gamma*self.gae_lambda*adv[-1]*np.array(dones[step])
                adv.append(advantage)
            adv.reverse()
            adv = np.array(adv[:-1])
            adv = T.tensor(adv, device=self.critic.device).unsqueeze(2)
            returns = adv + values.unsqueeze(2)
            adv = (adv - adv.mean()) / (adv.std()+1e-4)
        return adv, returns

    def learn(self, memory):
        # print("We are in the learn function")
        actor_states, states, actions, old_probs, rewards, actor_new_states, \
            states_, dones = memory.recall()
        device = self.critic.device
        state_arr = T.tensor(states, dtype=T.float, device=device)
        # print("State_arr shape: ", state_arr.shape)
        states__arr = T.tensor(states_, dtype=T.float, device=device)
        # print("New State_arr shape: ", states__arr.shape)
        r = T.tensor(rewards, dtype=T.float, device=device)
        # print("reward shape: ", r.shape)
        # print('actions: ', actions)
        # print('actions.shape: ', actions.shape)
        # print('actions[0]: ', actions[0])
        # action_arr = T.tensor(actions[self.agent_idx],
        #                       dtype=T.float, device=device)
        action_arr = T.tensor(actions,
                              dtype=T.float, device=device)
        # print("action_arr shape: ", action_arr.shape)
        # old_probs_arr = T.tensor(old_probs[self.agent_idx], dtype=T.float,
        #                          device=device)
        old_probs_arr = T.tensor(old_probs, dtype=T.float,
                                 device=device)
        # print("old_probs_arr shape: ", old_probs_arr.shape)
        # actor_states_arr = T.tensor(actor_states[self.agent_idx],
        #                             dtype=T.float, device=device)
        actor_states_arr = T.tensor(actor_states,
                                    dtype=T.float, device=device)
        # print("actor_states_arr shape: ", actor_states_arr.shape)
        adv, returns = self.calc_adv_and_returns((state_arr, states__arr,
                                                 r, dones))
        # print("old_probs_arr:", old_probs_arr)
        # print("actor_states_arr:", actor_states_arr)

        for epoch in range(self.n_epochs):
            batches = memory.generate_batches()
            for batch in batches:
                old_probs = old_probs_arr[batch]  # shape: [batch_size, 2]
                actions = action_arr[batch]  # shape: [batch_size, 2]
                actor_states = actor_states_arr[batch]  # shape: [batch_size, state_dim]
                dist = self.actor(actor_states)  # dist will have shape: [batch_size, 2, action_dim]
                new_probs = dist.log_prob(actions)  # shape: [batch_size, 2]

                # Calculate the probability ratio
                prob_ratio = T.exp(new_probs - old_probs)  # shape: [batch_size, 2]

                # Adjust adv for matching shapes
                adv_batch = adv[batch].squeeze(-1)  # This should be of shape [batch_size]

                weighted_probs = adv_batch.unsqueeze(1) * prob_ratio  # shape: [batch_size, 2]
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip,
                                                 1 + self.policy_clip) * adv_batch.unsqueeze(1)

                # Calculate entropy correctly
                entropy = dist.entropy().sum(dim=-1, keepdim=True)  # shape: [batch_size, 2]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()  # Ensure mean over batch
                actor_loss -= self.entropy_coefficient * entropy.mean()  # Average entropy loss over the batch

                # Optimizing actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                T.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor.optimizer.step()

                # Critic loss calculation remains unchanged
                states = state_arr[batch]
                critic_value = self.critic(states).squeeze()  # shape: [batch_size]
                critic_loss = (critic_value - returns[batch].squeeze()).pow(2).mean()
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()

        # for epoch in range(self.n_epochs):
        #     batches = memory.generate_batches()
        #     # print("batches:", batches)
        #     for batch in batches:
        #         # print("batch: ", batch)
        #         old_probs = old_probs_arr[batch]
        #         # print("old_probs_arr:", old_probs_arr)
        #         # print("action_arr:", action_arr)
        #         # print("old_probs.shape: ", old_probs.shape)
        #         actions = action_arr[batch]
        #         actor_states = actor_states_arr[batch]
        #         dist = self.actor(actor_states)
        #         new_probs = dist.log_prob(actions)
        #         # print("new_probs.shape: ", new_probs.shape)
        #         prob_ratio = T.exp(new_probs.sum(1, keepdims=True) - old_probs.
        #                            sum(1, keepdims=True))
        #         print("adv.shape: ",adv.shape)
        #         print("batch.shape: ", batch.shape)
        #         print("prob_ratio.shape: ", prob_ratio.shape)
        #         weighted_probs = adv[batch].squeeze(-1) * prob_ratio
        #         weighted_clipped_probs = T.clamp(
        #                 prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * \
        #             adv[batch].squeeze(-1)
        #         print("Probs shape: ", dist.probs.shape)
        #         entropy = dist.entropy().sum(2, keepdims=True)
        #         actor_loss = -T.min(weighted_probs,
        #                             weighted_clipped_probs)
        #         actor_loss -= self.entropy_coefficient * entropy
        #         self.actor.optimizer.zero_grad()
        #         actor_loss.mean().backward()
        #         T.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
        #         self.actor.optimizer.step()
        #
        #         states = state_arr[batch]
        #         critic_value = self.critic(states).squeeze()
        #         critic_loss = \
        #             (critic_value - returns[batch].squeeze()).pow(2).mean()
        #         self.critic.optimizer.zero_grad()
        #         critic_loss.backward()
        #         self.critic.optimizer.step()