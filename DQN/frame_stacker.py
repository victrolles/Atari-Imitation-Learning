from collections import deque

import numpy as np

class FrameStacker:
    def __init__(self, stack_size=2, frame_skip_size=1):
        self.stack_size = stack_size
        self.frame_skip_size = frame_skip_size
        self.size = (stack_size - 1)*frame_skip_size + 1
        self.frames = deque(maxlen=self.size)

    def reset(self, frame):
        """
        Reset frame stack with the first frame
        
        Args:
            frame (np.ndarray): The first frame.
        """

        self.frames.clear()
        for _ in range(self.size):
            self.frames.append(frame)

        return np.stack(list(self.frames)[: : self.frame_skip_size])

    def add(self, frame):
        """
        Add new frame and return stacked frames
        
        Args:
            frame (np.ndarray): The new frame.
        """
        self.frames.append(frame)
        return np.stack(list(self.frames)[: : self.frame_skip_size])

    def get(self):
        """
        Return stacked frames
        
        Returns:
            np.ndarray: The stacked frames.
        """
        return np.stack(list(self.frames)[: : self.frame_skip_size])