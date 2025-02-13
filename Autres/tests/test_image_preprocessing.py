import gymnasium as gym
import cv2
import numpy as np
import torch

# Initialize the environment
import ale_py
game_name = "MsPacman-v5" # "MsPacman-v5" or "Enduro-v5"
gym.register_envs(ale_py) 
env = gym.make(f"ALE/{game_name}", render_mode="rgb_array")
print(env.unwrapped.get_action_meanings())

# create a folder to save the images
import os
os.makedirs(f"Autres/images/{game_name}_2f", exist_ok=True)

def prepost_frame(frame, resize_size = 128, game_name = "MsPacman-v5"):
    """Prepost a frame.

    Args:
        frame (np.ndarray): The frame to prepost. Shape: (H, W, 3)
        image_size (int): The size of the image.
        game_name (str): The name of the game.
    """
    # Crop the image
    if game_name == "MsPacman-v5":
        cropped_frame = frame[1:171, 0:159]
    elif game_name == "Enduro-v5":
        cropped_frame = frame[0:154, 8:158]
    else:
        raise ValueError("Invalid game name")

    # Resize the image to 128x128
    resized_frame = cv2.resize(cropped_frame, (resize_size, resize_size))

    # Convert to grayscale
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # Normalize the image
    normalized_frame = gray_frame.astype(np.float32) / 255.0

    return normalized_frame

from collections import deque

class FrameStacker:
    def __init__(self, stack_size=4):
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)

    def reset(self, frame):
        """
        Reset frame stack with the first frame
        
        Args:
            frame (np.ndarray): The first frame.
        """

        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(frame)

    def add(self, frame):
        """
        Add new frame and return stacked frames
        
        Args:
            frame (np.ndarray): The new frame.
        """
        self.frames.append(frame)

    def get(self):
        """
        Return stacked frames
        
        Returns:
            np.ndarray: The stacked frames.
        """
        return np.stack(self.frames)


import matplotlib.pyplot as plt

def plot_stacked_frames(frames):
    """
    Plots 4 images in a 2x2 grid from a (4, 128, 128) tensor.
    
    Args:
        frames (numpy.ndarray or torch.Tensor): Shape (4, 128, 128), a stack of grayscale frames.
    """
    if isinstance(frames, torch.Tensor):
        frames = frames.numpy()  # Convert to numpy if it's a tensor

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))  # 2x2 grid
    for i, ax in enumerate(axes.flat):
        ax.imshow(frames[i], cmap='gray')  # Display grayscale image
        ax.axis('off')  # Hide axes for better visualization
        ax.set_title(f"Frame {i+1}")
    
    plt.tight_layout()
    plt.show()

# Initialize the frame stacker
frame_stacker = FrameStacker(stack_size=4)

# Reset the environment and get the first frame
state, _ = env.reset()
preprocessed_frame = prepost_frame(state, game_name = game_name)
frame_stacker.reset(preprocessed_frame)

for _ in range(70):
    state, _, _, _, _ = env.step(2)
    preprocessed_frame = prepost_frame(state, game_name = game_name)
    frame_stacker.add(preprocessed_frame)

# Plot the stacked frames
stack = frame_stacker.get()
print(f"Stack shape: {stack.shape}")
plot_stacked_frames(stack)

env.close()
