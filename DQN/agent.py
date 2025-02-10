import random

import torch

from DQN.dqn_model import DQNModel

class Agent():
    def __init__(self, obs_shape, num_actions, device):
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.device = device

        # Initialize the policy and target networks
        self.policy_net = DQNModel(obs_shape, num_actions).to(device)
        self.target_net = DQNModel(obs_shape, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.num_actions)  # Action aléatoire
        else:
            with torch.no_grad():
                state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
                return self.policy_net(state).argmax(dim=1).item()  # Action du réseau

