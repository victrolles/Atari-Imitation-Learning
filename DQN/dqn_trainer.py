import random

import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np

from DQN.config import BATCH_SIZE, GAMMA, LEARNING_RATE, ITER_PER_EPISODE
from DQN.replay_buffer import ReplayBuffer
from DQN.dqn_model import DQNModel

class DQNTrainer():

    def __init__(self, policy_net: DQNModel, target_net: DQNModel, replay_buffer: ReplayBuffer, device):
        self.policy_net = policy_net
        self.target_net = target_net
        self.replay_buffer = replay_buffer
        self.device = device

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()

    def train(self):
        list_loss = []
        list_q_values = []

        for _ in range(ITER_PER_EPISODE):
            data = self.replay_buffer.sample(BATCH_SIZE)
            
            # Compute Q values
            q_values = self.policy_net(data['states']).gather(1, data['actions']).squeeze(1)

            # Compute the expected Q values
            with torch.no_grad():
                next_q_values = self.target_net(data['next_states']).max(1)[0].detach()
                expected_q_values = data['rewards'].flatten() + (GAMMA * next_q_values * (1 - data['dones'].flatten()))

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