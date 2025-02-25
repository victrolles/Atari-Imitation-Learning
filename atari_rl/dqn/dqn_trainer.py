import random
import time

import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np

from atari_rl.rl.replay_buffer import ReplayBuffer
from atari_rl.dqn.dqn_model import DQNModel

class DQNTrainer():

    def __init__(self,
                 policy_net: DQNModel,
                 target_net: DQNModel,
                 replay_buffer: ReplayBuffer,
                 lr: float,
                 gamma: float,
                 batch_size: int,
                 iter_per_episode: int,
                 device: torch.device) -> None:
        
        self.policy_net = policy_net
        self.target_net = target_net
        self.replay_buffer = replay_buffer

        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.iter_per_episode = iter_per_episode

        self.device = device

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train(self):
        self.policy_net.train()
        self.target_net.eval()

        self.policy_net.to(self.device)
        self.target_net.to(self.device)

        list_loss = []
        list_q_values = []

        for _ in range(self.iter_per_episode):

            # Sample a batch of experiences
            data = self.replay_buffer.sample(self.batch_size)
            
            # Compute Q values
            q_values = self.policy_net(data['states']).gather(1, data['actions']).squeeze(1)

            # Compute the expected Q values
            with torch.no_grad():
                next_q_values = self.target_net(data['next_states']).max(1)[0].detach()
                expected_q_values = data['rewards'].flatten() + (self.gamma * next_q_values * (1 - data['dones'].flatten()))

            # Compute the loss
            loss = self.criterion(q_values, expected_q_values)

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Store statistics
            list_q_values.append(q_values.mean().item())
            list_loss.append(loss.item())

        return np.mean(list_loss), np.mean(list_q_values)