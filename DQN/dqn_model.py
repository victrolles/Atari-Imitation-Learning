import torch.nn as nn

class DQNModel(nn.Module):
    def __init__(self, obs_shape, num_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=4), # 1 canal d'entrée pour grayscale
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        return self.network(x)