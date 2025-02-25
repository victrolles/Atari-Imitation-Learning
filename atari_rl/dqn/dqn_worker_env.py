from collections import namedtuple

import gymnasium as gym
import ale_py
import torch
import numpy as np

from atari_rl.rl.utils import prepost_frame, scale_reward
from atari_rl.rl.frame_stacker import FrameStacker
from atari_rl.dqn.config import config
from atari_rl.dqn.utils import select_action

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))


class DqnWorkerEnv():

    def __init__(self, idx, training_id, queue, policy_net):
        self.config = config()
        self.idx = idx
        self.training_id = training_id
        self.queue = queue
        self.policy_net = policy_net

        self.frame_stacker = FrameStacker(stack_size=self.config.frame_stack_size,
                                          frame_skip_size=self.config.frame_skip_size)

        if torch.cuda.is_available():
            print("Training optimized with CUDA")
            self.device = torch.device("cuda")
        else:
            print("Training with CPU")
            self.device = torch.device("cpu")

        self.loop()

    def loop(self):
        print(f"Env {self.idx} is running")
        gym.register_envs(ale_py)
        env = gym.make(f"ALE/{self.config.game_name}", render_mode="rgb_array")
        env.reset()

        epsilon = self.config.epsilon_start
        idx = 0

        while True:
            
            idx += 1
            # Process the first frame
            frame, _ = env.reset()
            preprocessed_frame = prepost_frame(frame, self.config.image_size)
            stacked_preprocessed_frames = self.frame_stacker.reset(preprocessed_frame)

            # experiment on the environment to collect experiences
            for t in range(self.config.max_step_per_episode):
                # Select the action
                action = select_action(
                    self.policy_net,
                    self.device,
                    self.config.num_actions,
                    stacked_preprocessed_frames,
                    epsilon)

                # Perform the action
                next_frame, reward, done, truncated, _ = env.step(action)

                # Preprocess the next frame
                next_preprocessed_frame = prepost_frame(next_frame, self.config.image_size)
                next_stacked_preprocessed_frames = self.frame_stacker.add(next_preprocessed_frame)

                # Store the experience in the replay buffer

                self.queue.put(Experience(stacked_preprocessed_frames,
                                          np.array(action, dtype=np.int32),
                                          np.array(scale_reward(reward), dtype=np.float32),
                                          next_stacked_preprocessed_frames,
                                          np.array(done, dtype=np.float32)
                                         )
                            )
                
                stacked_preprocessed_frames = next_stacked_preprocessed_frames.copy()

                if done or truncated:
                    break

            self.writer.add_scalars(f"charts/episode_length", {self.idx: t}, idx)