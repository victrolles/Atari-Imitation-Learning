from collections import deque

import numpy as np

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

        return np.stack(self.frames)

    def add(self, frame):
        """
        Add new frame and return stacked frames
        
        Args:
            frame (np.ndarray): The new frame.
        """
        self.frames.append(frame)
        return np.stack(self.frames)

    def get(self):
        """
        Return stacked frames
        
        Returns:
            np.ndarray: The stacked frames.
        """
        return np.stack(self.frames)