import torch.optim as optim
import torch.nn as nn
import torch

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
        self.alpha = 0.1
        self.batch_size = batch_size
        self.iter_per_episode = iter_per_episode

        self.device = device

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train(self):
        stats = {
            "reward": 0,
            "loss": 0,
            "value_loss": 0,
            "value_loss2": 0,
            "q_values": 0
        }

        for _ in range(self.iter_per_episode):

            policy_batch = self.replay_buffer.sample(int(self.batch_size/2))
            expert_batch = self.expert_dataset.sample(int(self.batch_size/2), self.device, True, True)

            batch_state, batch_next_state, batch_action, batch_reward, batch_done, is_expert = get_concat_samples(policy_batch, expert_batch, self.device)

            # Notes: (our explanation of what iq loss is doing)
            # the loss takes in 2 points -> mer
            #       'reward' (calculated as current Q - expected value of next state)
            #       'value_loss' (calculated as current V - expected value of next state)
            # loss tries to minimize the difference between reward and value_loss
            
            current_q_values = self.policy_net(batch_state).gather(1, batch_action.unsqueeze(1))
            current_values = self.alpha * torch.logsumexp(self.policy_net(batch_state) / self.alpha, dim=1, keepdim=True)

            with torch.no_grad():
                next_q_values = self.alpha * torch.logsumexp(self.target_net(batch_next_state) / self.alpha, dim=1, keepdim=True)
                expected_q_values = (1 - batch_done) * self.gamma * next_q_values

                # # Apply clipping to prevent extreme values
                # expected_q_values = torch.clamp(expected_q_values, -10.0, 10.0)

            # Calculate 1st term of loss: -E_(ρ_expert)[Q(s, a) - γV(s')]
            reward = (current_q_values - expected_q_values)[is_expert]
            # reward_clipped = torch.clamp(reward, -10.0, 10.0)  # Prevent extreme values
            
            stats["reward"] += reward.mean().item() / self.iter_per_episode
            loss = -(reward).mean()
            stats["q_values"] += current_q_values.mean().item() / self.iter_per_episode
            stats["value_loss"] += loss / self.iter_per_episode

            # 2nd term for our loss (use expert and policy states): E_(ρ)[Q(s,a) - γV(s')]
            value_diff = (current_values - expected_q_values)
            # value_diff_clipped = torch.clamp(value_diff, -10.0, 10.0)  # Prevent extreme values
            value_loss = value_diff.mean()
            
            stats["value_loss2"] += value_loss.item() / self.iter_per_episode
            loss += value_loss

            stats["loss"] += loss.item() / self.iter_per_episode

            # Use χ2 divergence (adds a extra term to the loss)
            reward = (current_q_values - expected_q_values)
            chi2_loss = 1/(4 * self.alpha) * (reward**2).mean()
            loss += chi2_loss

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return stats