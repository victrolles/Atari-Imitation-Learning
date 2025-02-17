from dataclasses import dataclass

import gymnasium as gym
import ale_py
import numpy as np
import h5py
import torch

from atari_rl.rl.agent import Agent
from atari_rl.rl.utils import prepost_frame
from atari_rl.rl.frame_stacker import FrameStacker

NUM_EPISODES = 10

# Game parameters
GAME_NAME = "MsPacman-v5"
NUM_ACTIONS = 5
MAX_STEP_PER_EPISODE = 10000

# Agent parameters
IMAGE_SIZE = 84
FRAME_STACK_SIZE = 4
FRAME_SKIP_SIZE = 4
EPSILON = 0.05
USE_DETERMINISTIC = False
USE_EPSILON = False
TEMPERATURE = 1

@dataclass
class StateAction:
    state: np.ndarray
    action: int

class CreateExpertDatasetOnGym():

    def __init__(self):
        gym.register_envs(ale_py)
        self.env = gym.make(f"ALE/{GAME_NAME}", render_mode="rgb_array")
        self.env.reset()
        self.action = 0
        self.size = 0

        self.obs_shape = (FRAME_STACK_SIZE, IMAGE_SIZE, IMAGE_SIZE)  # On réduit les images pour accélérer l'entraînement
        self.num_actions = 5  # Nombre d'actions possibles
        print(f"Num actions: {self.num_actions}")

        if torch.cuda.is_available():
            print("Training optimized with CUDA")
            self.device = torch.device("cuda")
        else:
            print("Training with CPU")
            self.device = torch.device("cpu")

        self.agent = Agent(self.obs_shape, self.num_actions, self.device)
        self.frame_stacker = FrameStacker(stack_size=FRAME_STACK_SIZE, frame_skip_size=FRAME_SKIP_SIZE)

        self.h5_file = f"expert_dataset_{NUM_EPISODES}.h5"
        self.dataset_initialized = False

    def create_expert_dataset(self):
        for i in range(NUM_EPISODES):
            print(f"Episode {i}, size: {self.size}")
            list_state_action = self.run_one_episode()
            self.save_expert_dataset(list_state_action)
        self.env.close()

    def run_one_episode(self):
        list_state_action = []
        done = False

        frame, _ = self.env.reset()
        preprocessed_frame = prepost_frame(frame, IMAGE_SIZE)
        stacked_preprocessed_frames = self.frame_stacker.reset(preprocessed_frame)

        for t in range(MAX_STEP_PER_EPISODE):
            action = self.agent.select_action(stacked_preprocessed_frames, epsilon=0.05)

            next_frames, _, done, truncated, _ = self.env.step(action)
            next_preprocessed_frame = prepost_frame(next_frames, IMAGE_SIZE)
            next_stacked_preprocessed_frames = self.frame_stacker.add(next_preprocessed_frame)

            list_state_action.append(StateAction(stacked_preprocessed_frames, action))

            stacked_preprocessed_frames = next_stacked_preprocessed_frames.copy()

            if done or truncated:
                break
            
        self.size += len(list_state_action)
        return list_state_action
    
    def save_expert_dataset(self, list_state_action: list[StateAction]):
        """ Save dataset in an HDF5 file """
        with h5py.File(self.h5_file, "a") as h5f:
            if not self.dataset_initialized:
                num_samples = len(list_state_action)

                # Create datasets with maxshape=(None, ...)
                h5f.create_dataset("states", 
                                   shape=(num_samples,) + self.obs_shape, 
                                   maxshape=(None,) + self.obs_shape, 
                                   dtype=np.uint8)
                
                h5f.create_dataset("actions", 
                                   shape=(num_samples,), 
                                   maxshape=(None,), 
                                   dtype=np.int32)

                self.dataset_initialized = True  # Dataset is now created
            
            # Append new data
            num_existing = h5f["states"].shape[0]
            num_new = len(list_state_action)

            # Resize datasets to accommodate new data
            h5f["states"].resize((num_existing + num_new,) + self.obs_shape)
            h5f["actions"].resize((num_existing + num_new,))

            # Add new data
            for i, state_action in enumerate(list_state_action):
                h5f["states"][num_existing + i] = state_action.state
                h5f["actions"][num_existing + i] = state_action.action

if __name__ == "__main__":
    create_expert_dataset_on_gym = CreateExpertDatasetOnGym()
    create_expert_dataset_on_gym.create_expert_dataset()