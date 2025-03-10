from dataclasses import dataclass
import random
import os

import numpy as np
import h5py
import torch

class ExpertDataset:
    def __init__(self,
                 obs_shape: tuple,
                 num_actions: int,
                 expert_folder: str = "./results/datasets",
                 expert_name: str = "expert_dataset") -> None:
        
        self.expert_folder = expert_folder
        self.expert_name = expert_name
        self.expert_path = f"{self.expert_folder}/{self.expert_name}.h5"

        self.obs_shape = obs_shape
        self.num_actions = num_actions

        self.dataset_initialized = False
        self.size = 0

    def __len__(self):
        return self.size
        
    def create(self):
        """Initialize an empty dataset."""
        if not os.path.exists(self.expert_folder):
            os.makedirs(self.expert_folder)
            
        self.expert_path = f"{self.expert_folder}/{self.expert_name}.h5"
        with h5py.File(self.expert_path, "w") as h5f:
            h5f.create_dataset("states", shape=(0,) + self.obs_shape, maxshape=(None,) + self.obs_shape, dtype=np.float32)
            h5f.create_dataset("actions", shape=(0,), maxshape=(None,), dtype=np.int32)
            h5f.create_dataset("next_states", shape=(0,) + self.obs_shape, maxshape=(None,) + self.obs_shape, dtype=np.float32)
            h5f.create_dataset("dones", shape=(0,), maxshape=(None,), dtype=np.float32)
            self.dataset_initialized = True
    
    def add_batch(self,
            states: np.ndarray,
            actions: np.ndarray,
            next_states: np.ndarray,
            dones: np.ndarray) -> None:
        """Add a list of state-action pairs to the dataset."""
        if not self.dataset_initialized:
            self.create()

        with h5py.File(self.expert_path, "a") as h5f:
            num_existing = h5f["states"].shape[0]
            num_new = states.shape[0]
            
            h5f["states"].resize((num_existing + num_new,) + self.obs_shape)
            h5f["actions"].resize((num_existing + num_new,))
            h5f["next_states"].resize((num_existing + num_new,) + self.obs_shape)
            h5f["dones"].resize((num_existing + num_new,))

            h5f["states"][num_existing:] = states
            h5f["actions"][num_existing:] = actions
            h5f["next_states"][num_existing:] = next_states
            h5f["dones"][num_existing:] = dones

            self.size = h5f["states"].shape[0]

        # Renommer le fichier avec le nombre total d'éléments
        new_filename = f"{self.expert_folder}/{self.expert_name}_{self.size}.h5"
        os.rename(self.expert_path, new_filename)
        self.expert_path = new_filename

    def add(self,
            state: np.ndarray,
            action: np.ndarray,
            next_state: np.ndarray,
            done: np.ndarray) -> None:
        """Add a single state-action pair to the dataset."""
        if not self.dataset_initialized:
            self.create()

        with h5py.File(self.expert_path, "a") as h5f:
            num_existing = h5f["states"].shape[0]
            
            h5f["states"].resize((num_existing + 1,) + self.obs_shape)
            h5f["actions"].resize((num_existing + 1,))
            h5f["next_states"].resize((num_existing + 1,) + self.obs_shape)
            h5f["dones"].resize((num_existing + 1,))

            h5f["states"][num_existing] = state
            h5f["actions"][num_existing] = action
            h5f["next_states"][num_existing] = next_state
            h5f["dones"][num_existing] = done

            self.size = h5f["states"].shape[0]

        # Renommer le fichier avec le nombre total d'éléments
        new_filename = f"{self.expert_folder}/{self.expert_name}_{self.size}.h5"
        os.rename(self.expert_path, new_filename)
        self.expert_path = new_filename
    
    def load(self, size: int = None) -> dict:
        """Load all or part of the dataset."""
        with h5py.File(self.expert_path, "r") as h5f:
            if size is None or size > h5f["states"].shape[0]:
                size = h5f["states"].shape[0]
            
            states = h5f["states"][:size]
            actions = h5f["actions"][:size]
            next_states = h5f["next_states"][:size]
            dones = h5f["dones"][:size]

        return {
            "states": states,
            "actions": actions,
            "next_states": next_states,
            "dones": dones
        }
    
    def sample(self, batch_size: int, device: torch.device, to_torch = False) -> dict:
        """Sample a batch of state-action pairs randomly from the dataset."""
        with h5py.File(self.expert_path, "r") as h5f:

            num_samples = h5f["states"].shape[0]
            indices = sorted(random.sample(range(num_samples), batch_size))

            states = h5f["states"][indices]
            actions = h5f["actions"][indices]
            next_states = h5f["next_states"][indices]
            dones = h5f["dones"][indices]

            if to_torch:
                states = torch.tensor(states, dtype=torch.float32, device=device)
                actions = torch.tensor(actions, dtype=torch.long, device=device)
                next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
                dones = torch.tensor(dones, dtype=torch.float32, device=device)
            
        return {
            "states": states,
            "actions": actions,
            "next_states": next_states,
            "dones": dones
        }

# Example of use and verification  
if __name__ == "__main__":
    dataset = ExpertDataset((4, 4), 4)
    dataset.create()

    states = np.random.rand(10, 4, 4)
    actions = np.random.randint(0, 4, 10)
    next_states = np.random.rand(10, 4, 4)
    dones = np.random.rand(10)

    # print(f"shapes: {states}, {actions}, {next_states}, {dones}")
    print(f"shapes: {states.shape}, {actions.shape}, {next_states.shape}, {dones.shape}")

    dataset.add(states, actions, next_states, dones)

    # data = dataset.load()
    # print(data["states"], data["actions"], data["next_states"], data["dones"])

    data = dataset.sample(5, torch.device("cpu"), to_torch=True)
    print(data["states"].shape, data["actions"].shape, data["next_states"].shape, data["dones"].shape)
