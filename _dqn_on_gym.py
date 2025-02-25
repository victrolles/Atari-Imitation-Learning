import random
import time

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import ale_py
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from atari_rl.dqn.dqn_trainer import DQNTrainer
from atari_rl.rl.agent import Agent
from atari_rl.rl.utils import prepost_frame, scale_reward
from atari_rl.rl.frame_stacker import FrameStacker
from atari_rl.rl.replay_buffer import ReplayBuffer

# Game parameters
GAME_NAME = "MsPacman-v5"
RL_ALGORITHM = "DQN"
NUM_ACTIONS = 5

# Agent parameters
IMAGE_SIZE = 84
FRAME_STACK_SIZE = 4
FRAME_SKIP_SIZE = 4

# DQN parameters
GAMMA = 0.99
LEARNING_RATE = 1e-4
BUFFER_SIZE = 17000
BATCH_SIZE = 256
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.999
TARGET_UPDATE = 3
MAX_STEP_PER_EPISODE = 10000
ITER_PER_EPISODE = 175

NUM_EPISODES = 15000
USE_DETERMINISTIC = False
TEMPERATURE = 1

# Evaluation parameters
MODEL_NAME = "DQN_MsPacman-v5_290_10700.pt"
SAVE_MODEL = True
LOAD_MODEL = False
EVAL_RATE = 200

NUM_EPISODES_EVAL = 10
EPSILON_EVAL = 0.05
USE_DETERMINISTIC_EVAL = False
USE_EPSILON_EVAL = True
TEMPERATURE_EVAL = 1

class DQNOnGym():

    def __init__(self):
        gym.register_envs(ale_py)
        
        obs_shape = (FRAME_STACK_SIZE, IMAGE_SIZE, IMAGE_SIZE)
        print(f"Observation Shape : {obs_shape}, Num actions: {NUM_ACTIONS}")

        if torch.cuda.is_available():
            print("Training optimized with CUDA")
            self.device = torch.device("cuda")
        else:
            print("Training with CPU")
            self.device = torch.device("cpu")
        # self.device = torch.device("cpu")

        self.training_id = random.randint(0, 1000)
        print(f"Training ID: {self.training_id}")

        self.frame_stacker = FrameStacker(stack_size=FRAME_STACK_SIZE,
                                          frame_skip_size=FRAME_SKIP_SIZE)
        
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE,
                                          obs_shape,
                                          1,
                                          self.device)
        
        self.agent = Agent(obs_shape,
                           NUM_ACTIONS,
                           self.device)
        if LOAD_MODEL:
            self.agent.load_model("./results/models", MODEL_NAME)
        
        self.trainer = DQNTrainer(self.agent.policy_net,
                                  self.agent.target_net,
                                  self.replay_buffer,
                                  LEARNING_RATE,
                                  GAMMA,
                                  BATCH_SIZE,
                                  ITER_PER_EPISODE,
                                  self.device)

        self.writer = SummaryWriter(f"./results/tensorboard/{RL_ALGORITHM}_{GAME_NAME}_{self.training_id}")
        self.writer.add_text("Hyperparameters",f"BUFFER_SIZE: {BUFFER_SIZE}, BATCH_SIZE: {BATCH_SIZE}, LEARNING_RATE: {LEARNING_RATE}, GAMMA: {GAMMA}, EPSILON_START: {EPSILON_START}, EPSILON_END: {EPSILON_END}, EPSILON_DECAY: {EPSILON_DECAY}, TARGET_UPDATE: {TARGET_UPDATE}, MAX_STEP_PER_EPISODE: {MAX_STEP_PER_EPISODE}, ITER_PER_EPISODE: {ITER_PER_EPISODE}, NUM_EPISODES: {NUM_EPISODES}, USE_DETERMINISTIC: {USE_DETERMINISTIC}, TEMPERATURE: {TEMPERATURE}, EVAL_RATE: {EVAL_RATE}, NUM_EPISODES_EVAL: {NUM_EPISODES_EVAL}, EPSILON_EVAL: {EPSILON_EVAL}, USE_DETERMINISTIC_EVAL: {USE_DETERMINISTIC_EVAL}, USE_EPSILON_EVAL: {USE_EPSILON_EVAL}, TEMPERATURE_EVAL: {TEMPERATURE_EVAL}")
    
    def train_loop(self):
        env = gym.make(f"ALE/{GAME_NAME}", render_mode="rgb_array")
        env.reset()
        env_step = 0
        training_iter = 0
        total_reward = 0
        epsilon = EPSILON_START
        list_time = []
        delta_time = time.time()
        for episode in range(NUM_EPISODES):
            list_time.append(time.time() - delta_time)
            print(f"Episode {episode}, epsilon: {epsilon:.3f}, total_reward: {int(total_reward)}, mean time: {np.mean(list_time):.3f}")
            delta_time = time.time()
            total_reward = 0

            # delta_time = time.time()

            # Process the first frame
            frame, _ = env.reset()
            preprocessed_frame = prepost_frame(frame, IMAGE_SIZE)
            stacked_preprocessed_frames = self.frame_stacker.reset(preprocessed_frame)

            # print(f"Preprocessing time: {time.time() - delta_time:.3f}")
            # delta_time = time.time()

            # experiment on the environment to collect experiences
            for t in range(MAX_STEP_PER_EPISODE):
                # Select the action
                action = self.agent.select_action(stacked_preprocessed_frames,
                                                  epsilon,
                                                  )

                # Perform the action
                next_frame, reward, done, truncated, _ = env.step(action)
                total_reward += reward

                # Preprocess the next frame
                next_preprocessed_frame = prepost_frame(next_frame, IMAGE_SIZE)
                next_stacked_preprocessed_frames = self.frame_stacker.add(next_preprocessed_frame)

                # Store the experience in the replay buffer

                self.replay_buffer.add(stacked_preprocessed_frames,
                                       np.array(action, dtype=np.int32),
                                       np.array(scale_reward(reward), dtype=np.float32),
                                       next_stacked_preprocessed_frames,
                                       np.array(done, dtype=np.float32))
                stacked_preprocessed_frames = next_stacked_preprocessed_frames.copy()

                if done or truncated:
                    break
            
            # print(f"Experience time: {time.time() - delta_time:.3f}")
            # delta_time = time.time()

            # Train the model
            mean_loss, mean_q_value = self.trainer.train()
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
            if episode % TARGET_UPDATE == 0:
                self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())

            # print(f"Training time: {time.time() - delta_time:.3f}")
            # delta_time = time.time()

            # Log the results
            env_step += t
            training_iter += ITER_PER_EPISODE

            self.writer.add_scalar("charts/episode_length", t, episode)
            self.writer.add_scalar("charts/total_reward", total_reward, episode)
            self.writer.add_scalar("charts/env_step", env_step, episode)
            self.writer.add_scalar("charts/training_iter", training_iter, episode)

            self.writer.add_scalar("training/epsilon", epsilon, episode)
            self.writer.add_scalar("training/mean_loss", mean_loss, episode)
            self.writer.add_scalar("training/mean_q_value", mean_q_value, episode)
            self.writer.add_scalar("training/buffer_size", len(self.replay_buffer), episode)

            # print(f"Logging time: {time.time() - delta_time:.3f}")

            if episode % EVAL_RATE == 0:
                self.eval_loop(episode)

            
                 

        self.writer.close()
        env.close()

    def eval_loop(self, train_episode: int):

        # Create paths
        video_path = f"./results/videos/{RL_ALGORITHM}_{GAME_NAME}_{self.training_id}/episode_{train_episode}"
        model_path = f"./results/models/{RL_ALGORITHM}_{GAME_NAME}_{self.training_id}/"
        model_name = f"episode_{train_episode}"

        # save the model
        if SAVE_MODEL:
            self.agent.save_model(model_path, model_name)

        # Evaluate the model
        env_eval = gym.make(f"ALE/{GAME_NAME}", render_mode="rgb_array")
        env_eval = RecordVideo(env_eval, video_folder=video_path)
        env_eval.reset()

        list_reward = []
        list_length = []
        total_reward = 0

        for episode in range(NUM_EPISODES_EVAL):
            print(f"(Eval) Episode {episode}, total_reward: {int(total_reward)}")
            total_reward = 0

            # Process the first frame
            frame, _ = env_eval.reset()
            preprocessed_frame = prepost_frame(frame, IMAGE_SIZE)
            stacked_preprocessed_frames = self.frame_stacker.reset(preprocessed_frame)

            # experiment on the environment to collect experiences
            for t in range(MAX_STEP_PER_EPISODE):
                # Select the action
                action = self.agent.select_action(stacked_preprocessed_frames,
                                                  EPSILON_EVAL,
                                                  training=USE_EPSILON_EVAL,
                                                  deterministic=USE_DETERMINISTIC_EVAL,
                                                  temperature=TEMPERATURE_EVAL)

                # Perform the action
                next_frame, reward, done, truncated, _ = env_eval.step(action)
                total_reward += reward

                # Preprocess the next frame
                next_preprocessed_frame = prepost_frame(next_frame, IMAGE_SIZE)
                next_stacked_preprocessed_frames = self.frame_stacker.add(next_preprocessed_frame)
                
                stacked_preprocessed_frames = next_stacked_preprocessed_frames.copy()

                if done or truncated:
                    break

            list_reward.append(total_reward)
            list_length.append(t)

        self.writer.add_scalar("eval/eval_mean_reward", np.mean(list_reward), train_episode)
        self.writer.add_scalar("eval/eval_mean_length", np.mean(list_length), train_episode)

        env_eval.close()

if __name__ == '__main__':
    dqn_on_gym = DQNOnGym()
    dqn_on_gym.train_loop()