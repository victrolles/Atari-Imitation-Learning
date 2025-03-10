import random

import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np

from atari_rl.rl.replay_buffer import ReplayBuffer
from atari_rl.dqn.dqn_model import DQNModel
from atari_rl.iql.expert_dataset import ExpertDataset
from atari_rl.iql.utils import get_concat_samples

class IQLTrainer():

    def __init__(self,
                 policy_net: DQNModel,
                 target_net: DQNModel,
                 replay_buffer: ReplayBuffer,
                 expert_dataset: ExpertDataset,
                 lr: float,
                 gamma: float,
                 batch_size: int,
                 iter_per_episode: int,
                 device: torch.device) -> None:
        
        self.policy_net = policy_net
        self.target_net = target_net

        self.replay_buffer = replay_buffer
        self.expert_dataset = expert_dataset

        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.iter_per_episode = iter_per_episode

        self.device = device

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train(self):
        list_1st_term_loss = []
        list_2nd_term_loss = []
        list_chi2_loss = []
        list_q_values = []

        for _ in range(self.iter_per_episode):

            policy_batch = self.replay_buffer.sample(self.batch_size)
            expert_batch = self.expert_dataset.sample(self.batch_size)

            batch_state, batch_next_state, batch_action, batch_reward, batch_done, is_expert = get_concat_samples(policy_batch, expert_batch, self.device)

            # Calculate 1st term of loss: -E_(ρ_expert)[Q(s, a) - γV(s')]
            q_values = self.policy_net(batch_state).gather(1, batch_action).squeeze(1)

            # Compute the expected Q values
            with torch.no_grad():
                next_q_values = self.target_net(batch_next_state)
                expected_q_values = self.gamma * next_q_values * (1 - batch_done.flatten())

            # Compute the loss
            reward = (q_values - expected_q_values)[is_expert]
            loss = -(reward).mean()

            # 2nd term for our loss (use expert and policy states): E_(ρ)[Q(s,a) - γV(s')]
            value_loss = (self.policy_net(batch_state) - expected_q_values).mean()
            loss += value_loss

            # Use χ2 divergence (adds a extra term to the loss)
            chi2_loss = 1/(4 * 0.5) * (reward**2).mean()
            loss += chi2_loss

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Store statistics
            list_q_values.append(q_values.mean().item())
            list_1st_term_loss.append(reward.mean().item())
            list_2nd_term_loss.append(value_loss.item())
            list_chi2_loss.append(chi2_loss.item())

        return np.mean(list_1st_term_loss), np.mean(list_2nd_term_loss), np.mean(list_chi2_loss), np.mean(list_q_values)