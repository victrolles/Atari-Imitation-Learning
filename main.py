import random

import gymnasium as gym
import ale_py
import torch
from torch.utils.tensorboard import SummaryWriter

from DQN.config import EPSILON_START, EPSILON_DECAY, EPSILON_END, NUM_EPISODES, TARGET_UPDATE, STEP_PER_UPDATE
from DQN.dqn_trainer import DQNTrainer
from DQN.agent import Agent
from DQN.utils import prepost_image_state

class RlOnEnduro():

    def __init__(self):
        gym.register_envs(ale_py)
        self.env = gym.make("ALE/Enduro-v5", render_mode="rgb_array")
        self.env.reset()
        self.action = 0

        self.obs_shape = (128, 128)  # On réduit les images pour accélérer l'entraînement
        self.num_actions = self.env.action_space.n  # Nombre d'actions possibles
        print(f"Num actions: {self.num_actions}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.agent = Agent(self.obs_shape, self.num_actions, self.device)
        self.trainer = DQNTrainer(self.agent.policy_net, self.agent.target_net, self.device)

        self.writer = SummaryWriter(f"tensorboard/DQN_Enduro_{random.randint(0, 100)}")
        self.writer.add_text("Hyperparameters", f"EPSILON_START: {EPSILON_START}, EPSILON_DECAY: {EPSILON_DECAY}, EPSILON_END: {EPSILON_END}, NUM_EPISODES: {NUM_EPISODES}, TARGET_UPDATE: {TARGET_UPDATE}, STEP_PER_UPDATE: {STEP_PER_UPDATE}")


    def train_loop(self):
        epsilon = EPSILON_START
        for episode in range(NUM_EPISODES):
            print(f"Episode {episode}, epsilon: {epsilon:.3f}")
            state, _ = self.env.reset()
            state = prepost_image_state(state)
            total_reward = 0

            for t in range(STEP_PER_UPDATE):
                action = self.agent.select_action(state, epsilon)
                next_state, reward, done, truncated, _ = self.env.step(action)
                next_state = prepost_image_state(next_state)
                self.trainer.experience_memory._append((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward

                if done or truncated:
                    break

            loss = self.trainer.train()  # Entraînement du réseau
            self.writer.add_scalar("charts/epsilon", epsilon, episode)
            self.writer.add_scalar("charts/loss", loss, episode)
            self.writer.add_scalar("charts/total_reward", total_reward, episode)

            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)  # Réduction epsilon

            if episode % TARGET_UPDATE == 0:
                self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())  # Mise à jour du réseau cible

        self.writer.close()
        self.env.close()

    

if __name__ == '__main__':
    rl_on_enduro = RlOnEnduro()
    rl_on_enduro.train_loop()