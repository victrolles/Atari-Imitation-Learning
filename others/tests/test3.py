import torch
import numpy as np
import random
import time

class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.position = 0
        self.full = False

        # Préallocation des arrays pour éviter les copies répétées
        self.states = np.zeros((capacity, *state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)  # Actions discrètes
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.bool_)

    def add(self, state, action, reward, next_state, done):
        """Ajoute une transition dans le buffer, écrasant les plus anciennes."""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done

        # Mise à jour de la position (FIFO)
        self.position = (self.position + 1) % self.capacity
        if self.position == 0:
            self.full = True

    def sample(self, batch_size):
        """Retourne un batch aléatoire sous forme de tensors PyTorch."""
        max_index = self.capacity if self.full else self.position
        indices = np.random.choice(max_index, batch_size, replace=False)

        batch = (
            torch.tensor(self.states[indices], dtype=torch.float32),
            torch.tensor(self.actions[indices], dtype=torch.int64),
            torch.tensor(self.rewards[indices], dtype=torch.float32),
            torch.tensor(self.next_states[indices], dtype=torch.float32),
            torch.tensor(self.dones[indices], dtype=torch.bool),
        )
        return batch

    def __len__(self):
        """Retourne la taille actuelle du buffer."""
        return self.capacity if self.full else self.position

if __name__ == "__main__":

    # Exemple d'utilisation
    # state_dim = (4,)
    state_dim = (4, 84, 84)
    buffer = ReplayBuffer(capacity=100_000, state_dim=state_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ajout d'expériences fictives
    for _ in range(20_000):
        state = np.random.rand(*state_dim)
        action = np.random.randint(0, 2)
        reward = np.random.uniform(-1, 1)
        next_state = np.random.rand(*state_dim)
        done = np.random.choice([False, True])
        buffer.add(state, action, reward, next_state, done)

    print(f"Buffer size: {len(buffer)}")

    delta_time = time.time()
    list_time = []
    list_time2 = []

    # Extraction d'un batch
    for i in range(624):
        delta2 = time.time()
        batch = buffer.sample(32)
        list_time.append(time.time() - delta2)
        delta_time3 = time.time()
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = batch
        batch_states = batch_states.to(device)
        batch_actions = batch_actions.to(device)
        batch_rewards = batch_rewards.to(device)
        batch_next_states = batch_next_states.to(device)
        batch_dones = batch_dones.to(device)
        list_time2.append(time.time() - delta_time3)
    print(f"Batch extraction time: {time.time() - delta_time:.3f}")
    print(f"Mean time: {np.mean(list_time):.6f}")
    print(f"Mean time2: {np.mean(list_time2):.6f}")
    print("idx:", i)
