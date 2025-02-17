import torch.nn as nn

from atari_rl.rl.utils import compute_output_size

class DQNModel(nn.Module):
    def __init__(self, obs_shape, num_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 8, stride=4), # 1 canal d'entrée pour grayscale
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(compute_output_size(obs_shape[1]), 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        return self.network(x)