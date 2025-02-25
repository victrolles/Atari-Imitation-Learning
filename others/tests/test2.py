import torch
import numpy as np
import random
import time
from torch.utils.data import Dataset, DataLoader

class RLDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        """Retourne une seule expérience sous forme de tensor PyTorch"""
        return (torch.tensor(self.buffer.states[idx], dtype=torch.float32),
                torch.tensor(self.buffer.actions[idx], dtype=torch.int64),
                torch.tensor(self.buffer.rewards[idx], dtype=torch.float32),
                torch.tensor(self.buffer.next_states[idx], dtype=torch.float32),
                torch.tensor(self.buffer.dones[idx], dtype=torch.bool))

class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.position = 0
        self.full = False

        # Allocation des données en NumPy
        self.states = np.zeros((capacity, *state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.bool_)

    def add(self, state, action, reward, next_state, done):
        """Ajoute une expérience au buffer."""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done

        # Gestion FIFO
        self.position = (self.position + 1) % self.capacity
        if self.position == 0:
            self.full = True

    def __len__(self):
        return self.capacity if self.full else self.position
    
if __name__ == "__main__":

    # ⚡ Initialisation du buffer
    # state_dim = (4,)
    state_dim = (4, 84, 84)
    buffer = ReplayBuffer(capacity=100_000, state_dim=state_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ajout d'expériences fictives
    for _ in range(10_000):
        state = np.random.rand(*state_dim)
        action = np.random.randint(0, 2)
        reward = np.random.uniform(-1, 1)
        next_state = np.random.rand(*state_dim)
        done = np.random.choice([False, True])
        buffer.add(state, action, reward, next_state, done)

    # ⚡ Création du DataLoader avec préchargement et multithreading
    num_workers = 12
    delta_time = time.time()
    dataset = RLDataset(buffer)
    # dataloader = DataLoader(dataset,
    #                         batch_size=256,
    #                         shuffle=True,
    #                         num_workers=num_workers,
    #                         pin_memory=True,
    #                         prefetch_factor=2,
    #                         persistent_workers=True)
    dataloader = DataLoader(dataset,
                            batch_size=1024,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=False,
                            persistent_workers=False)

    print("num_workers:", num_workers)
    print(f"DataLoader creation time: {time.time() - delta_time:.3f}")
    delta_time = time.time()
    list_time2 = []

    # Test d'extraction de batch

    for idx, batch in enumerate(dataloader):
        delta_time3 = time.time()
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = batch
        batch_states = batch_states.to(device)
        batch_actions = batch_actions.to(device)
        batch_rewards = batch_rewards.to(device)
        batch_next_states = batch_next_states.to(device)
        batch_dones = batch_dones.to(device)
        list_time2.append(time.time() - delta_time3)

    print(f"Batch extraction time: {time.time() - delta_time:.3f}")
    print("mean time:", np.mean(list_time2))
    print("idx:", idx)
