from dataclasses import dataclass
import random
import os

import numpy as np
import h5py
import torch

@dataclass
class Experience:
    state: np.ndarray
    action: int
    next_state: np.ndarray
    done: bool

class ExpertDataset:
    def __init__(self,
                 obs_shape: tuple,
                 num_actions: int,
                 expert_folder: str = "./results/datasets",
                 expert_name: str = "iql_expert_dataset") -> None:
        
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
    
    def add(self, list_experience: list[Experience]) -> None:
        """Add a list of state-action pairs to the dataset."""
        if not self.dataset_initialized:
            self.create()

        with h5py.File(self.expert_path, "a") as h5f:
            num_existing = h5f["states"].shape[0]
            num_new = len(list_experience)
            
            h5f["states"].resize((num_existing + num_new,) + self.obs_shape)
            h5f["actions"].resize((num_existing + num_new,))
            h5f["next_states"].resize((num_existing + num_new,) + self.obs_shape)
            h5f["dones"].resize((num_existing + num_new,))

            for i, exp in enumerate(list_experience):
                h5f["states"][num_existing + i] = exp.state
                h5f["actions"][num_existing + i] = exp.action
                h5f["next_states"][num_existing + i] = exp.next_state
                h5f["dones"][num_existing + i] = exp.done

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
    
    def load_actions(self, size: int = None) -> np.ndarray:
        """Load all or part of the actions from the dataset."""
        with h5py.File(self.expert_path, "r") as h5f:
            if size is None or size > h5f["actions"].shape[0]:
                size = h5f["actions"].shape[0]
            
            actions = h5f["actions"][:size]
            
        return actions
    
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
    dataset = ExpertDataset((2, 2), 4)
    dataset.create()

    states = np.random.rand(5, 2, 2)
    actions = np.random.randint(0, 4, 5)
    next_states = np.random.rand(5, 2, 2)
    dones = np.random.rand(5)

    experiences = [Experience(states[i], actions[i], next_states[i], dones[i]) for i in range(5)]
    dataset.add(experiences)

    load = dataset.load()
    
    sample = dataset.sample(2, torch.device("cpu"), to_torch=True)

    print("Inputs :")
    for idx, experience in enumerate(experiences):
        print(f"Experience {idx} : {experience}")

    print("\nLoad :")
    for key, value in load.items():
        print(f"{key} : {value}")

    print("\nSample :")
    for key, value in sample.items():
        print(f"{key} : {value}")