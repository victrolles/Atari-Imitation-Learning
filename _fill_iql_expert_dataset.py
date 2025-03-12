import random

import gymnasium as gym
import ale_py
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from atari_rl.iql.expert_dataset import ExpertDataset, Experience
from atari_rl.iql.utils import balance_experience_dataset
from atari_rl.rl.agent import Agent
from atari_rl.rl.utils import prepost_frame
from atari_rl.rl.frame_stacker import FrameStacker

NUM_EPISODES = 100

# Game parameters
GAME_NAME = "Freeway-v5"
RL_ALGORITHM = "FILL_IQL"
NUM_ACTIONS = 3
MAX_STEP_PER_EPISODE = 10000

# Agent parameters
MODEL_PATH = "./results_saved"
MODEL_NAME = "DQN_Freeway-v5_8/episode_15250"
IMAGE_SIZE = 84
FRAME_STACK_SIZE = 4
FRAME_SKIP_SIZE = 4
EPSILON = 0.05
DETERMINISTIC = True
USE_EPSILON = True
TEMPERATURE = 1

# Expert dataset parameters
H5_NAME = "iql_expert_dataset"
H5_PATH = "./results/datasets"

class Main():

    def __init__(self):
        gym.register_envs(ale_py)
        self.env = gym.make(f"ALE/{GAME_NAME}", render_mode="rgb_array")
        self.env.reset()
        self.action = 0
        self.size = 0

        obs_shape = (FRAME_STACK_SIZE, IMAGE_SIZE, IMAGE_SIZE)  # On réduit les images pour accélérer l'entraînement
        print(f"Observation Shape : {obs_shape}, Num actions: {NUM_ACTIONS}")

        if torch.cuda.is_available():
            print("Training optimized with CUDA")
            self.device = torch.device("cuda")
        else:
            print("Training with CPU")
            self.device = torch.device("cpu")

        self.training_id = random.randint(0, 1000)
        print(f"Filling number : {self.training_id}")
        h5_name = f"{H5_NAME}_{self.training_id}"

        self.agent = Agent(obs_shape, NUM_ACTIONS, self.device)
        self.agent.load_model(MODEL_PATH, MODEL_NAME)
        self.frame_stacker = FrameStacker(stack_size=FRAME_STACK_SIZE, frame_skip_size=FRAME_SKIP_SIZE)
        self.expert_dataset = ExpertDataset(obs_shape,
                                            NUM_ACTIONS,
                                            expert_folder=H5_PATH,
                                            expert_name=h5_name)
        
        self.writer = SummaryWriter(f"./results/tensorboard/{RL_ALGORITHM}_{GAME_NAME}_{self.training_id}")
        self.writer.add_text("Hyperparameters", f"Game: {GAME_NAME}, Num actions: {NUM_ACTIONS}, Max step per episode: {MAX_STEP_PER_EPISODE}")

    def fill_dataset(self):
        
        for i in range(NUM_EPISODES):
            print(f"Episode {i}, size: {len(self.expert_dataset)}")
            list_experience = []
            done = False
            total_reward = 0

            frame, _ = self.env.reset()
            preprocessed_frame = prepost_frame(frame, IMAGE_SIZE)
            stacked_preprocessed_frames = self.frame_stacker.reset(preprocessed_frame)

            for _ in range(MAX_STEP_PER_EPISODE):
                action = self.agent.select_action(stacked_preprocessed_frames,
                                                epsilon=EPSILON,
                                                deterministic=DETERMINISTIC,
                                                training=USE_EPSILON,
                                                temperature=TEMPERATURE)

                next_frames, reward, done, truncated, _ = self.env.step(action)
                total_reward += reward
                next_preprocessed_frame = prepost_frame(next_frames, IMAGE_SIZE)
                next_stacked_preprocessed_frames = self.frame_stacker.add(next_preprocessed_frame)

                list_experience.append(Experience(stacked_preprocessed_frames, action, next_stacked_preprocessed_frames, done))

                stacked_preprocessed_frames = next_stacked_preprocessed_frames.copy()

                if done or truncated:
                    break

            self.writer.add_scalar("charts/rewards", total_reward, i)
            self.writer.add_scalar("charts/size", len(self.expert_dataset), i)

            balanced_list_experience = balance_experience_dataset(list_experience)
            self.expert_dataset.add(balanced_list_experience)

        self.env.close()

if __name__ == "__main__":
    main = Main()
    main.fill_dataset()