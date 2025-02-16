import numpy as np
from DQN.replay_buffer import ReplayBuffer

# verify the replay buffer
# python -m test2.py

if __name__ == "__main__":
    max_capacity = 4
    state_dim = (2,2)
    action_dim = 1
    device = "cpu"

    replay_buffer = ReplayBuffer(max_capacity, state_dim, action_dim, device)

    for i in range(8):
        print("Iteration: ", i)
        state = np.random.rand(2)
        action = np.random.randint(1)
        reward = np.random.rand(1)
        next_state = np.random.rand(2)
        done = np.random.randint(1)

        replay_buffer.add(state, action, reward, next_state, done)
        print(replay_buffer.sample(min(replay_buffer.size, 4))['rewards'])


    print("Replay buffer test passed")
