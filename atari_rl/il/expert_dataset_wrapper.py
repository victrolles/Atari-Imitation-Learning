import torch
from torch.utils.data import Dataset

from atari_rl.il.expert_dataset import StateAction

class ExpertDatasetWrapper(Dataset):
    """Custom Dataset wrapper for expert data."""
    def __init__(self, expert_data: list[StateAction]) -> None:
        self.expert_data = expert_data

    def __len__(self):
        return len(self.expert_data)

    def __getitem__(self, idx):
        sa = self.expert_data[idx]
        state = torch.tensor(sa.state, dtype=torch.float32)
        action = torch.tensor(sa.action, dtype=torch.long)
        return state, action