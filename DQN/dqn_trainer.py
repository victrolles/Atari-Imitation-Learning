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
        self.optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()

    def train(self):
        while len(self.experience_memory) > BATCH_SIZE:
            batch = self.experience_memory.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = batch

            states = torch.tensor(np.array(states), device=self.device, dtype=torch.float32)
            actions = torch.tensor(actions, device=self.device, dtype=torch.int64).unsqueeze(1)
            rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32).unsqueeze(1)
            next_states = torch.tensor(np.array(next_states), device=self.device, dtype=torch.float32)
            dones = torch.tensor(dones, device=self.device, dtype=torch.float32).unsqueeze(1)

            q_values = self.policy_net(states).gather(1, actions)
            next_q_values = self.target_net(next_states).max(1)[0]

            states = states.unsqueeze(1)
            next_states = next_states.unsqueeze(1)

            expected_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

            loss = self.loss_fn(q_values, expected_q_values.detach())

            loss_value = loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss_value