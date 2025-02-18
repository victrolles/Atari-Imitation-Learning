import gymnasium as gym
import ale_py
import torch

class ILOnGym():

    def __init__(self):
        gym.register_envs(ale_py)

    def train_loop(self):
        pass

    def eval_loop(self):
        pass

if __name__ == "__main__":
    il_on_gym = ILOnGym()
    il_on_gym.train_loop()
        