import random
import os

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import ale_py
import torch
from torch.utils.tensorboard import SummaryWriter

from DQN.config import EPSILON_START, EPSILON_DECAY, EPSILON_END, NUM_EPISODES, TARGET_UPDATE, STEP_PER_UPDATE, GAME_NAME, RL_ALGORITHM, IMAGE_SIZE, FRAME_STACK_SIZE, SAVE_RATE, FRAME_SKIP_SIZE
from DQN.dqn_trainer import DQNTrainer
from DQN.agent import Agent
from DQN.utils import prepost_frame, scale_reward
from DQN.frame_stacker import FrameStacker

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
        
        self.agent = Agent(self.training_id, self.obs_shape, self.num_actions, self.device)
        self.trainer = DQNTrainer(self.agent.policy_net, self.agent.target_net, self.device)

        self.writer = SummaryWriter(f"./tensorboard/{RL_ALGORITHM}_{GAME_NAME}_{self.training_id}")
        self.writer.add_text("Hyperparameters", f"EPSILON_START: {EPSILON_START}, EPSILON_DECAY: {EPSILON_DECAY}, EPSILON_END: {EPSILON_END}, NUM_EPISODES: {NUM_EPISODES}, TARGET_UPDATE: {TARGET_UPDATE}, STEP_PER_UPDATE: {STEP_PER_UPDATE}")


    def train_loop(self):
        epsilon = EPSILON_START
        for episode in range(NUM_EPISODES):
            print(f"Episode {episode}, epsilon: {epsilon:.3f}")

            frame, _ = self.env.reset()
            preprocessed_frame = prepost_frame(frame, IMAGE_SIZE)
            stacked_preprocessed_frames = self.frame_stacker.reset(preprocessed_frame)
            
            total_reward = 0

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

            for t in range(STEP_PER_UPDATE):
                action = self.agent.select_action(stacked_preprocessed_frames, epsilon)
                next_frame, reward, done, truncated, _ = self.env.step(action)
                next_preprocessed_frame = prepost_frame(next_frame, IMAGE_SIZE)
                next_stacked_preprocessed_frames = self.frame_stacker.add(next_preprocessed_frame)

                self.trainer.experience_memory._append((stacked_preprocessed_frames, action, scale_reward(reward), next_stacked_preprocessed_frames, done))
                stacked_preprocessed_frames = next_stacked_preprocessed_frames
                total_reward += reward

                if done or truncated:
                    break

            mean_loss, mean_q_value = self.trainer.train()  # Entraînement du réseau
            self.writer.add_scalar("charts/epsilon", epsilon, episode)
            self.writer.add_scalar("charts/total_reward", total_reward, episode)
            self.writer.add_scalar("charts/episode_length", t, episode)

            self.writer.add_scalar("losses/mean_loss", mean_loss, episode)
            self.writer.add_scalar("losses/mean_q_value", mean_q_value, episode)

            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)  # Réduction epsilon

            if episode % TARGET_UPDATE == 0:
                self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())  # Mise à jour du réseau cible

            # Si un enregistrement vidéo était en cours, on ferme proprement
            

        self.writer.close()
        self.env.close()

if __name__ == '__main__':
    rl_on_gym = RlOnGym()
    rl_on_gym.train_loop()