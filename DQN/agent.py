import random

import torch

from DQN.dqn_model import DQNModel
from DQN.config import LOAD_MODEL, SAVE_MODEL, SAVE_PATH, RL_ALGORITHM, GAME_NAME

class Agent():
    def __init__(self, training_id, obs_shape, num_actions, device):
        self.training_id = training_id
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.device = device

        # Initialize the policy and target networks
        self.policy_net = DQNModel(obs_shape, num_actions).to(device)
        self.target_net = DQNModel(obs_shape, num_actions).to(device)
        self.load_model()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.num_actions)  # Action aléatoire
        else:
            with torch.no_grad():
                state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
                return self.policy_net(state).argmax(dim=1).item()  # Action du réseau
            
    def save_model(self, episode):
        if SAVE_MODEL:
            torch.save(self.policy_net.state_dict(), f"models/{RL_ALGORITHM}_{GAME_NAME}_{self.training_id}_{episode}.pt")
            print(f"Model saved at episode {episode}")

    def load_model(self):
        if LOAD_MODEL:
            self.policy_net.load_state_dict(torch.load(SAVE_PATH))
            self.policy_net.eval()
            print(f"Model loaded from {SAVE_PATH}")

