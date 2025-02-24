import torch
from torch.utils.data import Dataset

class RLDataset(Dataset):
    def __init__(self, states, actions, rewards, next_states, dones):
        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.long)
        self.rewards = torch.tensor(rewards, dtype=torch.float32)
        self.next_states = torch.tensor(next_states, dtype=torch.float32)
        self.dones = torch.tensor(dones, dtype=torch.bool)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx], self.dones[idx]