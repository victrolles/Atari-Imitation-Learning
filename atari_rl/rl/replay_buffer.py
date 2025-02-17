import numpy as np
from numpy._typing import _ShapeLike
import torch

class ReplayBuffer:
    def __init__(self, max_capacity: int, state_dim: _ShapeLike, action_dim: _ShapeLike, device: torch.device):
        self.max_capacity = max_capacity
        self.device = device
        self.index = 0
        self.size = 0
        
        # Initialize memory as numpy arrays
        self.states = np.zeros((max_capacity, *state_dim), dtype=np.float32)
        self.actions = np.zeros((max_capacity, action_dim), dtype=np.int32)
        self.rewards = np.zeros((max_capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((max_capacity, *state_dim), dtype=np.float32)
        self.dones = np.zeros((max_capacity, 1), dtype=np.float32)

    def add(self,
            state: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            next_state: np.ndarray,
            done: np.ndarray) -> None:
        """
        Store a new experience in the buffer (FIFO mechanism).

        Experiences are stored in the buffer as numpy arrays rather than tensors to optimize memory usage.

        Args:
            state: The current state.
            action: The action taken in the current state.
            reward: The reward received after taking the action.
            next_state: The next state after taking the action.
            done: A flag indicating if the episode is done
        """
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done
        
        # Update the index
        self.index = (self.index + 1) % self.max_capacity
        self.size = min(self.size + 1, self.max_capacity)

    def sample(self, batch_size: int) -> dict:
        """
        Sample a batch of experiences randomly.

        Args:
            batch_size: The number of experiences to sample.

        Returns:
            A dictionary containing the sampled batch of experiences. (states, actions, rewards, next_states, dones)
        """
        indices = np.random.choice(self.size, batch_size, replace=False)
        return {
        "states": torch.tensor(self.states[indices], dtype=torch.float32, device=self.device),
        "actions": torch.tensor(self.actions[indices], dtype=torch.long, device=self.device),
        "rewards": torch.tensor(self.rewards[indices], dtype=torch.float32, device=self.device),
        "next_states": torch.tensor(self.next_states[indices], dtype=torch.float32, device=self.device),
        "dones": torch.tensor(self.dones[indices], dtype=torch.float32, device=self.device),
    }

    def __len__(self):
        """Return the number of stored samples."""
        return self.size
