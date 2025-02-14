import random

import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np

from DQN.config import BATCH_SIZE, GAMMA, LEARNING_RATE, BUFFER_SIZE
from DQN.experience_memory import ExperienceMemory

class DQNTrainer():

    def __init__(self, policy_net, target_net, device):
        self.policy_net = policy_net
        self.target_net = target_net
        self.device = device

        self.experience_memory = ExperienceMemory(BUFFER_SIZE)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()

    def train(self):
        list_loss = []
        list_q_values = []

        while len(self.experience_memory) > BATCH_SIZE:
            batch = self.experience_memory.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = batch

            # Convert to tensors
            states = torch.tensor(np.array(states), device=self.device, dtype=torch.float32)
            actions = torch.tensor(actions, device=self.device, dtype=torch.int64)
            rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
            next_states = torch.tensor(np.array(next_states), device=self.device, dtype=torch.float32)
            dones = torch.tensor(dones, device=self.device, dtype=torch.float32)
            
            # Compute Q values
            q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            list_q_values.append(q_values.mean().item())

            # Compute the expected Q values
            with torch.no_grad():
                next_q_values = self.target_net(next_states).max(1)[0].detach()
                expected_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

            # Compute the loss
            loss = self.criterion(q_values, expected_q_values)
            list_loss.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return np.mean(list_loss), np.mean(list_q_values)