from MAPPO_agent import Agent

class MAPPO:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions,
                 alpha=1e-4, gamma=0.95, chkpt_dir='tmp/mappo/'):
        self.agents = []
        self.n_agents = n_agents

        # Initialize each agent for the environment
        for agent_idx in range(n_agents):  # Loop through the number of agents directly
            self.agents.append(Agent(actor_dims, critic_dims,  # Pass single values
                               n_actions, agent_idx,gamma=gamma,
                               alpha=alpha, chkpt_dir=chkpt_dir))

        # print("self.agents: ", self.agents)

    def save_checkpoint(self):
        # Save model checkpoints for each agent
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        # Load model checkpoints for each agent
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs):
        # For each agent, select an action and corresponding probabilities
        actions = {}
        probs = {}
        for agent_idx in range(self.n_agents):  # Loop through the number of agents
            action, prob = self.agents[agent_idx].choose_action(raw_obs)  # Use the same observation for all
            actions[agent_idx] = action
            probs[agent_idx] = prob
        return actions, probs

    def learn(self, memory):
        # Each agent uses the shared memory to learn
        actor_states, states, actions, probs, rewards, actor_new_states, new_states, dones = memory.recall()

        # Iterate through each agent and allow them to update their policies
        for agent in self.agents:
            agent.learn(memory)
