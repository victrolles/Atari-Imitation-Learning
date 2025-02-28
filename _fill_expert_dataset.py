import gymnasium as gym
import ale_py
import torch

from atari_rl.il.expert_dataset import StateAction, ExpertDataset
from atari_rl.rl.agent import Agent
from atari_rl.rl.utils import prepost_frame
from atari_rl.rl.frame_stacker import FrameStacker

NUM_EPISODES = 100

# Game parameters
GAME_NAME = "Freeway-v5"
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
H5_NAME = "expert_dataset"
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

        self.agent = Agent(obs_shape, NUM_ACTIONS, self.device)
        self.agent.load_model(MODEL_PATH, MODEL_NAME)
        self.frame_stacker = FrameStacker(stack_size=FRAME_STACK_SIZE, frame_skip_size=FRAME_SKIP_SIZE)
        self.expert_dataset = ExpertDataset(obs_shape,
                                            NUM_ACTIONS,
                                            expert_folder=H5_PATH,
                                            expert_name=H5_NAME)

    def fill_dataset(self):
        
        for i in range(NUM_EPISODES):
            print(f"Episode {i}, size: {len(self.expert_dataset)}")
            list_state_action = []
            done = False

            frame, _ = self.env.reset()
            preprocessed_frame = prepost_frame(frame, IMAGE_SIZE)
            stacked_preprocessed_frames = self.frame_stacker.reset(preprocessed_frame)

            for _ in range(MAX_STEP_PER_EPISODE):
                action = self.agent.select_action(stacked_preprocessed_frames,
                                                epsilon=EPSILON,
                                                deterministic=DETERMINISTIC,
                                                training=USE_EPSILON,
                                                temperature=TEMPERATURE)

                next_frames, _, done, truncated, _ = self.env.step(action)
                next_preprocessed_frame = prepost_frame(next_frames, IMAGE_SIZE)
                next_stacked_preprocessed_frames = self.frame_stacker.add(next_preprocessed_frame)

                list_state_action.append(StateAction(stacked_preprocessed_frames, action))

                stacked_preprocessed_frames = next_stacked_preprocessed_frames.copy()

                if done or truncated:
                    break

            self.expert_dataset.add(list_state_action)

        self.env.close()

if __name__ == "__main__":
    main = Main()
    main.fill_dataset()