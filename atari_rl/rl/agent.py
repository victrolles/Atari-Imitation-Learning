import numpy as np
import random
import os

import torch

from atari_rl.dqn.dqn_model import DQNModel

class Agent():
    def __init__(self, obs_shape, num_actions: int, device: torch.device):
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.device = device

        # Initialize the policy and target networks
        self.policy_net = DQNModel(obs_shape, num_actions).to(device)
        self.target_net = DQNModel(obs_shape, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def select_action(self,
                      state: np.ndarray,
                      epsilon: float = 0.0,
                      training: bool = True,
                      deterministic: bool = True,
                      temperature: float = 1) -> int:
        
        """
        Select an action using epsilon-greedy strategy

        :param state: state of the environment
        :param epsilon: exploration rate (0.0 -> 1.0)
        :param training: whether the agent is training or not
        :param deterministic: whether to select the action deterministically or not
        :return: action index
        """

        if training and random.random() < epsilon:
            return random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
                output = self.policy_net(state)

                if deterministic:
                    output = torch.argmax(output, dim=1)
                else:
                    output = torch.multinomial(torch.softmax(output / temperature, dim=1), 1)

                return int(output.cpu().item())

            
    def save_model(self, model_path=None, model_name=None):
        
            if not os.path.exists(model_path):
                os.makedirs(model_path)

            model_full_path = f"{model_path}/{model_name}.pt"
            print(f"Model saved at {model_full_path}")

            torch.save(self.policy_net.state_dict(), model_full_path)
            

    def load_model(self, model_path=None, model_name=None):
            
            model_full_path = f"{model_path}/{model_name}.pt"
            print(f"Model loaded from {model_full_path}")

            self.policy_net.load_state_dict(torch.load(model_path))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()