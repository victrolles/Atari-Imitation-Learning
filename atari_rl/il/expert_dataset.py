from dataclasses import dataclass
import random
import os

import numpy as np
import h5py


@dataclass
class StateAction:
    state: np.ndarray
    action: int

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
            self.dataset_initialized = True
    
    def add(self, list_state_action: list[StateAction]):
        """Add a list of state-action pairs to the dataset."""
        if not self.dataset_initialized:
            self.create()

        with h5py.File(self.expert_path, "a") as h5f:
            num_existing = h5f["states"].shape[0]
            num_new = len(list_state_action)
            
            h5f["states"].resize((num_existing + num_new,) + self.obs_shape)
            h5f["actions"].resize((num_existing + num_new,))
            
            for i, sa in enumerate(list_state_action):
                h5f["states"][num_existing + i] = sa.state
                h5f["actions"][num_existing + i] = sa.action

            self.size = h5f["states"].shape[0]

        # Renommer le fichier avec le nombre total d'éléments
        new_filename = f"{self.expert_folder}/{self.expert_name}_{self.size}.h5"
        os.rename(self.expert_path, new_filename)
        self.expert_path = new_filename
    
    def load(self, size: int = None) -> list[StateAction]:
        """Load all or part of the dataset."""
        with h5py.File(self.expert_path, "r") as h5f:
            if size is None or size > h5f["states"].shape[0]:
                size = h5f["states"].shape[0]
            
            states = h5f["states"][:size]
            actions = h5f["actions"][:size]
            
        return [StateAction(state, action) for state, action in zip(states, actions)]
    
    def load_actions(self, size: int = None) -> np.ndarray:
        """Load all or part of the actions from the dataset."""
        with h5py.File(self.expert_path, "r") as h5f:
            if size is None or size > h5f["actions"].shape[0]:
                size = h5f["actions"].shape[0]
            
            actions = h5f["actions"][:size]
            
        return actions
    
    def sample(self, batch_size: int) -> list[StateAction]:
        """Sample a batch of state-action pairs randomly from the dataset."""
        with h5py.File(self.expert_path, "r") as h5f:
            num_samples = h5f["states"].shape[0]
            indices = sorted(random.sample(range(num_samples), batch_size))
            states = h5f["states"][indices]
            actions = h5f["actions"][indices]
            
        return [StateAction(state, action) for state, action in zip(states, actions)]

# Example of use and verification  
if __name__ == "__main__":
    dataset = ExpertDataset((4, 4), 4)
    dataset.create()
    
    list_state_action = [StateAction(np.random.randint(0, 255, (4, 4)), 2) for _ in range(5)]
    dataset.add(list_state_action)
    print("list state action")
    print(list_state_action)
    
    print("dataset")
    print(len(dataset))
    print(dataset.load())
    print(dataset.sample(3))
