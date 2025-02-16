import random
import os

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import ale_py
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from DQN.config import EPSILON_START, EPSILON_DECAY, EPSILON_END, NUM_EPISODES, TARGET_UPDATE, MAX_STEP_PER_EPISODE, GAME_NAME, RL_ALGORITHM, IMAGE_SIZE, FRAME_STACK_SIZE, SAVE_RATE, FRAME_SKIP_SIZE, BUFFER_SIZE, ITER_PER_EPISODE
from DQN.dqn_trainer import DQNTrainer
from DQN.agent import Agent
from DQN.utils import prepost_frame, scale_reward
from DQN.frame_stacker import FrameStacker
from DQN.replay_buffer import ReplayBuffer

class RlOnGym():

    def __init__(self):
        gym.register_envs(ale_py)
        self.env = gym.make(f"ALE/{GAME_NAME}", render_mode="rgb_array")
        self.env.reset()
        self.action = 0

        self.obs_shape = (FRAME_STACK_SIZE, IMAGE_SIZE, IMAGE_SIZE)  # On réduit les images pour accélérer l'entraînement
        self.num_actions = 5  # Nombre d'actions possibles
        print(f"Num actions: {self.num_actions}")

        if torch.cuda.is_available():
            print("Training optimized with CUDA")
            self.device = torch.device("cuda")
        else:
            print("Training with CPU")
            self.device = torch.device("cpu")

        self.training_id = random.randint(0, 1000)
        print(f"Training ID: {self.training_id}")

        self.frame_stacker = FrameStacker(stack_size=FRAME_STACK_SIZE, frame_skip_size=FRAME_SKIP_SIZE)
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE, self.obs_shape, 1, self.device)
        
        self.agent = Agent(self.training_id, self.obs_shape, self.num_actions, self.device)
        self.trainer = DQNTrainer(self.agent.policy_net, self.agent.target_net, self.replay_buffer, self.device)

        self.writer = SummaryWriter(f"./tensorboard/{RL_ALGORITHM}_{GAME_NAME}_{self.training_id}")
        self.writer.add_text("Hyperparameters", f"EPSILON_START: {EPSILON_START}, EPSILON_DECAY: {EPSILON_DECAY}, EPSILON_END: {EPSILON_END}, NUM_EPISODES: {NUM_EPISODES}, TARGET_UPDATE: {TARGET_UPDATE}, MAX_STEP_PER_EPISODE: {MAX_STEP_PER_EPISODE}")

    def train_loop(self):
        env_step = 0
        training_iter = 0
        epsilon = EPSILON_START
        for episode in range(NUM_EPISODES):
            print(f"Episode {episode}, epsilon: {epsilon:.3f}")

            # Process the first frame
            frame, _ = self.env.reset()
            preprocessed_frame = prepost_frame(frame, IMAGE_SIZE)
            stacked_preprocessed_frames = self.frame_stacker.reset(preprocessed_frame)

            self.record_and_save(episode)

            # experiment on the environment to collect experiences
            for t in range(MAX_STEP_PER_EPISODE):
                # Select the action
                action = self.agent.select_action(stacked_preprocessed_frames, epsilon)

                # Perform the action
                next_frame, reward, done, truncated, _ = self.env.step(action)

                # Preprocess the next frame
                next_preprocessed_frame = prepost_frame(next_frame, IMAGE_SIZE)
                next_stacked_preprocessed_frames = self.frame_stacker.add(next_preprocessed_frame)

                self.replay_buffer.add(stacked_preprocessed_frames,
                                       np.array(action, dtype=np.int32),
                                       np.array(scale_reward(reward), dtype=np.float32),
                                       next_stacked_preprocessed_frames,
                                       np.array(done, dtype=np.float32))
                
                stacked_preprocessed_frames = next_stacked_preprocessed_frames.copy()

                if done or truncated:
                    break

            # Train the model
            mean_loss, mean_q_value = self.trainer.train()
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
            if episode % TARGET_UPDATE == 0:
                self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())

            # Log the results
            env_step += t
            training_iter += ITER_PER_EPISODE

            self.writer.add_scalar("charts/epsilon", epsilon, episode)
            self.writer.add_scalar("charts/episode_length", t, episode)
            self.writer.add_scalar("losses/env_step", env_step, episode)

            self.writer.add_scalar("losses/mean_loss", mean_loss, episode)
            self.writer.add_scalar("losses/mean_q_value", mean_q_value, episode)
            self.writer.add_scalar("losses/buffer_size", len(self.replay_buffer), episode)
            self.writer.add_scalar("losses/training_iter", training_iter, episode)     

        self.writer.close()
        self.env.close()

    def record_and_save(self, episode):
        if episode != 0 and episode % SAVE_RATE == 0:
            video_path = f"./videos/{RL_ALGORITHM}_{GAME_NAME}_{self.training_id}_episode_{episode}"
            self.env = RecordVideo(self.env, video_folder=video_path, episode_trigger=lambda x: True)
            self.agent.save_model(episode)
        elif episode != 2 and (episode-2) % SAVE_RATE == 0:
            self.env.close()
            self.env = gym.make(f"ALE/{GAME_NAME}", render_mode="rgb_array")
            self.env.reset()

            path_to_video_to_remove = f"./videos/{RL_ALGORITHM}_{GAME_NAME}_{self.training_id}_episode_{episode-2}/rl-video-episode-1.mp4"
            if os.path.exists(path_to_video_to_remove):
                os.remove(path_to_video_to_remove)

if __name__ == '__main__':
    rl_on_gym = RlOnGym()
    rl_on_gym.train_loop()