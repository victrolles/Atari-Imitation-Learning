import gymnasium as gym
import ale_py
import numpy as np
import multiprocessing
import time

def play_randomly(process_id, num_episodes=20):
    """Plays random actions in the MsPacman-v4 environment."""
    gym.register_envs(ale_py)
    env = gym.make("ALE/MsPacman-v5", render_mode=None)  # Set to "human" to render
    list_time = []
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        delta_time = time.time()
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        list_time.append(time.time() - delta_time)
        print(f"Process {process_id}, Episode {episode + 1}: Total Reward = {total_reward}, mean time: {np.mean(list_time):.3f}")
    env.close()

def main(num_processes=4, num_episodes=5):
    """Runs multiple processes to play Pac-Man randomly."""
    processes = []
    for i in range(num_processes):
        p = multiprocessing.Process(target=play_randomly, args=(i, num_episodes))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

if __name__ == "__main__":
    num_processes = 12
    num_episodes = 200
    main(num_processes, num_episodes)
