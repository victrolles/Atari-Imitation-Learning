import multiprocessing
from _dqn_on_gym import DQNOnGym

def train_agent(process_id):
    """Trains a DQN agent on the MsPacman-v4 environment."""
    dqn_on_gym = DQNOnGym()
    dqn_on_gym.train_loop()
    print(f"Process {process_id} finished training.")

def main(num_processes=4):
    """Runs multiple processes to train DQN agents in parallel."""
    processes = []
    for i in range(num_processes):
        p = multiprocessing.Process(target=train_agent, args=(i,))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

if __name__ == "__main__":
    num_processes =11
    main(num_processes)
