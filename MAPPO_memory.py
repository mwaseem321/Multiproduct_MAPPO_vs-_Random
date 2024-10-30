import numpy as np

class PPOMemory:
    def __init__(self, batch_size, T, n_agents,
                 critic_dims, actor_dims, n_actions):

        # Initialize memory buffers for critic and actor networks
        self.states = np.zeros((T, n_agents, critic_dims), dtype=np.float32)
        self.rewards = np.zeros((T, 1), dtype=np.float32) # single global reward
        self.dones = np.zeros((T, 1), dtype=np.float32)
        self.new_states = np.zeros((T, n_agents, critic_dims), dtype=np.float32)

        # Memory buffers for each agent's actor network
        self.actor_states = np.zeros((T, n_agents, actor_dims), dtype=np.float32)
        self.actor_new_states = np.zeros((T, n_agents, actor_dims), dtype=np.float32)
        # self.actions = np.zeros((T, n_procs, n_actions), dtype=np.float32)
        # self.actions = np.zeros((T, n_agents, n_actions), dtype=np.float32)
        self.actions = np.zeros((T, n_agents), dtype=int)
        self.probs = np.zeros((T, n_agents), dtype=np.float32)

        self.mem_cntr = 0
        self.n_states = T
        self.critic_dims = critic_dims
        self.actor_dims = actor_dims
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.batch_size = batch_size

    def recall(self):
        # Recall stored memory for all agents
        return self.actor_states, \
            self.states, \
            self.actions, \
            self.probs, \
            self.rewards, \
            self.actor_new_states, \
            self.new_states, \
            self.dones

    def generate_batches(self):
        # Generate shuffled batches for training
        n_batches = int(self.n_states // self.batch_size)
        indices = np.arange(self.n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i * self.batch_size:(i + 1) * self.batch_size]
                   for i in range(n_batches)]
        return batches

    def store_memory(self, raw_obs, state, action, probs, reward,
                     raw_obs_, state_, done):
        index = self.mem_cntr % self.n_states
        # print("index: ", index)
        # Store the critic network's state and reward data
        self.states[index] = state
        self.new_states[index] = state_
        self.dones[index] = done
        self.rewards[index] = reward
        # print("raw_obs: ", raw_obs)
        # print("state: ", state)
        # print("action: ", action)
        # print("probs: ", probs)
        # print("reward: ", reward)
        # print("raw_obs_: ", raw_obs_)
        # print("state_: ", state_)
        # print("done: ", done)
        # Store the actor network's data for each agent
        # print("self.actions: ",self.actions )
        # print("self.actor_states.shape:", self.actor_states.shape)
        # print("self.actions.shape:", self.actions.shape)
        # print('action:', action)
        # print("self.actions[index]: ", self.actions[index])
        # print("self.actions[index][0]:", self.actions[index][0])
        # print("self.actions[index][1]:", self.actions[index][1])
        # Convert actions and probs dictionaries to arrays for each agent
        actions_arr = np.array([action[i] for i in range(self.n_agents)])
        probs_arr = np.array([probs[i] for i in range(self.n_agents)])

        self.actions[index] = actions_arr  # Action array should be of shape (T, n_agents)
        self.actor_states[index] = raw_obs  # Observed state (actor) for each agent (T, n_agents, actor_dims)
        self.actor_new_states[index] = raw_obs_  # Next observed state (actor) for each agent
        # print("probs_arr.shape:", probs_arr.shape)
        self.probs[index] = probs_arr  # Probability distribution for each agent (T, n_agents, n_actions)


        # for i in range(self.n_agents):
        #     # print("i", i)
        #     # print('index: ', index)
        #     # print('action: ', action)
        #     # print("action[i]: ", action[i])
        #     # print("self.actions[index][i]: ", self.actions[index][i])
        #     # print("self.actions[index]: ", self.actions[index])
        #     self.actions[index][i] = action[i]  # Use index to access the correct slot
        #     # print("raw_obs: ", raw_obs)
        #     # print("raw_obs: ", raw_obs[i])
        #     # print("self.actor_states[i][index]: ", self.actor_states[index][i])
        #     # print("self.actor_states[i]: ", self.actor_states[i])
        #     self.actor_states[index][i] = raw_obs[i]
        #     self.actor_new_states[index][i] = raw_obs_[i]
        #     self.probs[index][i] = probs[i]
        self.mem_cntr += 1

    def clear_memory(self):
        # Reset all memory buffers to zero
        self.states = np.zeros((self.n_states, self.n_agents, self.critic_dims),
                               dtype=np.float32)
        self.rewards = np.zeros((self.n_states, 1),
                                dtype=np.float32)
        self.dones = np.zeros((self.n_states, self.n_agents), dtype=np.float32)
        self.new_states = np.zeros((self.n_states, self.n_agents,
                                    self.critic_dims), dtype=np.float32)

        # Reset the actor-specific data for all agents
        self.actor_states = np.zeros((self.n_states, self.n_agents, self.actor_dims),
                                      dtype=np.float32)
        self.actor_new_states = np.zeros((self.n_states, self.n_agents, self.actor_dims),
                                          dtype=np.float32)
        # self.actions = np.zeros((self.n_states, self.n_agents, self.n_actions),
        #                          dtype=np.float32)
        self.actions = np.zeros((self.n_states, self.n_agents),
                                dtype=int)
        self.probs = np.zeros((self.n_states, self.n_agents),
                              dtype=np.float32)
