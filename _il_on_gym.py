import random

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import ale_py
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np

from atari_rl.rl.agent import Agent
from atari_rl.rl.utils import prepost_frame
from atari_rl.rl.frame_stacker import FrameStacker
from atari_rl.il.expert_dataset import ExpertDataset
from atari_rl.il.expert_dataset_wrapper import ExpertDatasetWrapper

# Game parameters
GAME_NAME = "MsPacman-v5"
RL_ALGORITHM = "IL"
NUM_ACTIONS = 5

# Agent parameters
IMAGE_SIZE = 84
FRAME_STACK_SIZE = 4
FRAME_SKIP_SIZE = 4

# Imitation Learning parameters
LEARNING_RATE = 1e-4
EXPERT_NAME = "expert_dataset_21668"
VAL_LOADED_SIZE = 1000
LOADED_SIZE = 6400
BATCH_SIZE = 32
EPOCHS = 200

# Evaluation parameters
MODEL_NAME = "DQN_MsPacman-v5_290_10700.pt"
SAVE_MODEL = True
LOAD_MODEL = False

MAX_STEP_PER_EPISODE = 10000
NUM_EPISODES_EVAL = 10
EPSILON_EVAL = 0.05
USE_DETERMINISTIC_EVAL = True
USE_EPSILON_EVAL = True
TEMPERATURE_EVAL = 1

class ILOnGym():

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

        self.training_id = random.randint(0, 1000)
        print(f"Training ID: {self.training_id}")

        self.expert_dataset = ExpertDataset(obs_shape, NUM_ACTIONS, expert_name=EXPERT_NAME)

        self.frame_stacker = FrameStacker(stack_size=FRAME_STACK_SIZE,
                                          frame_skip_size=FRAME_SKIP_SIZE)
        
        self.agent = Agent(obs_shape,
                           NUM_ACTIONS,
                           self.device)
        if LOAD_MODEL:
            self.agent.load_model("./results/models", MODEL_NAME)

        # Loss function
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.agent.policy_net.parameters(), LEARNING_RATE)

        self.writer = SummaryWriter(f"./results/tensorboard/{RL_ALGORITHM}_{GAME_NAME}_{self.training_id}")
        self.writer.add_graph("Hyperparameters", {"Learning rate": LEARNING_RATE,
                                                  "Batch size": BATCH_SIZE,
                                                  "Loaded size": LOADED_SIZE,
                                                  "Validation loaded size": VAL_LOADED_SIZE,
                                                  "Epochs": EPOCHS})

    def loop(self):

        for epoch in range(EPOCHS):
            print(f"Epoch {epoch}")

            self.train()
            self.validation()

            if epoch % 10 == 0:
                self.eval(epoch)

    def train(self):
        self.agent.policy_net.train()
        list_loss = []

        # Load portion of the expert dataset
        dfsample = self.expert_dataset.sample(LOADED_SIZE)

        # Create DataLoader
        dataset = ExpertDatasetWrapper(dfsample)
        dataloader = DataLoader(dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                num_workers=4)
        
        for idx, data in enumerate(dataloader):
            print(f"{idx}/{len(dataloader)}", end="\r")

            self.optimizer.zero_grad()
        
            inputs = data[0].to(self.device)
            labels = data[1].to(self.device)

            outputs = self.agent.policy_net(inputs)
            loss = self.criterion(outputs, labels.long())

            loss.backward()
            self.optimizer.step()

            list_loss.append(loss.item())

        mean_loss = np.mean(list_loss)
        self.writer.add_scalar("Loss", mean_loss)

    def validation(self):
        self.agent.policy_net.eval()

        total_correct = 0
        total_samples = 0

        # Load portion of the expert dataset
        dfsample = self.expert_dataset.sample(VAL_LOADED_SIZE)

        # Create DataLoader
        dataset = ExpertDatasetWrapper(dfsample)
        dataloader = DataLoader(dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                num_workers=4)
        
        for idx, data in enumerate(dataloader):
            print(f"{idx}/{len(dataloader)}", end="\r")

            inputs = data[0].to(self.device)
            labels = data[1].to(self.device)

            outputs = self.agent.policy_net(inputs)
            outputs = torch.argmax(outputs, dim=1)
            total_samples += labels.size(0)
            total_correct += (outputs == labels).sum().item()

        accuracy = total_correct / total_samples
        self.writer.add_scalar("Validation accuracy", accuracy)

    def eval(self, train_episode: int):

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

        self.writer.add_scalar("Eval reward", np.mean(list_reward))
        self.writer.add_scalar("Eval length", np.mean(list_length))

        env_eval.close()

if __name__ == "__main__":
    il_on_gym = ILOnGym()
    il_on_gym.loop()
        