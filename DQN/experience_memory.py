from collections import deque

import numpy as np

class ExperienceMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def _append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        
        # Remove sampled elements from buffer
        for idx in sorted(indices, reverse=True):  
            del self.buffer[idx]  

        states, actions, rewards, dones, next_states = zip(*samples)
        return states, actions, rewards, dones, next_states