import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class DiscreteActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, # n_actions:  all possible actions
                 fc1_dims=128, fc2_dims=128, chkpt_dir='models/'):
        super(DiscreteActorNetwork, self).__init__()

        # Set up directory for saving models

        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_discrete_ppo')

        # Define neural network layers
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.action_logits = nn.Linear(fc2_dims, n_actions)

        # Optimizer for the actor network
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        # Use GPU if available
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # Ensure input state is on the correct device
        x = T.tanh(self.fc1(state))
        x = T.tanh(self.fc2(x))

        # Get action logits for discrete action space
        action_logits = self.action_logits(x)

        # Create a Categorical distribution for the discrete actions
        dist = Categorical(logits=action_logits)
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha,
                 fc1_dims=128, fc2_dims=128, chkpt_dir='models/'):
        super(CriticNetwork, self).__init__()

        # Set up directory for saving models

        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_ppo')

        # Define neural network layers for the value function
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.value = nn.Linear(fc2_dims, 1)

        # Optimizer for the critic network
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        # Use GPU if available
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # Ensure input state is on the correct device
        x = T.tanh(self.fc1(state))
        x = T.tanh(self.fc2(x))

        # Calculate the value of the current state
        value = self.value(x)
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
