import random
import time
from collections import namedtuple

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from atari_rl.dqn.dqn_trainer import DQNTrainer
from atari_rl.rl.replay_buffer import ReplayBuffer
from atari_rl.dqn.config import config
from atari_rl.dqn.dqn_model import DQNModel

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class DqnWorkerTrainer():

    def __init__(self, idx, training_id, queue, policy_net):
        self.config = config()
        self.idx = idx
        self.training_id = training_id
        self.queue = queue
        self.policy_net = policy_net
        self.writer = writer
        self.target_net = DQNModel(self.config.obs_shape, self.config.num_actions)

        if torch.cuda.is_available():
            print("Training optimized with CUDA")
            self.device = torch.device("cuda")
        else:
            print("Training with CPU")
            self.device = torch.device("cpu")

        self.replay_buffer = ReplayBuffer(
            self.config.buffer_size,
            self.config.obs_shape,
            1,
            self.device)

        self.trainer = DQNTrainer(
            policy_net=self.policy_net,
            target_net=self.target_net,
            replay_buffer=self.replay_buffer,
            lr=self.config.learning_rate,
            gamma=self.config.gamma,
            batch_size=self.config.batch_size,
            iter_per_episode=self.config.iter_per_episode,
            device=self.device
        )

        self.writer = SummaryWriter(f"./results/tensorboard/{self.config.rl_algorithm}_{self.config.game_name}_{training_id}_trainer")
        hyperparams = "\n".join([f"**{key}**: {value}" for key, value in vars(config).items() if not key.startswith("__")])
        self.writer.add_text("Hyperparameters", hyperparams)

        self.loop()

    def loop(self):
        print(f"Trainer {self.idx} is running")
        idx = 0
        while True:
            self.empty_queue()
        
            if len(self.replay_buffer) > self.config.batch_size:
                print(f"Trainer {self.idx}, epoch {idx}, replay buffer size {len(self.replay_buffer)}")
                if self.idx == 0:
                    self.writer.add_scalar("charts/replay_buffer", len(self.replay_buffer), idx)
                idx += 1
                self.trainer.train()

    def empty_queue(self):
        while self.queue.qsize() > 0:
            experience: Experience = self.queue.get()
            self.replay_buffer.add(experience.state,
                                   experience.action,
                                   experience.reward,
                                   experience.next_state,
                                   experience.done)