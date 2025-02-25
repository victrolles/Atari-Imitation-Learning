import multiprocessing as mp
import random

from torch.utils.tensorboard import SummaryWriter

from atari_rl.dqn.dqn_worker_env import DqnWorkerEnv as Env
from atari_rl.dqn.dqn_worker_trainer import DqnWorkerTrainer as Trainer
from atari_rl.dqn.dqn_model import DQNModel
from atari_rl.dqn.config import config

def main():

    training_id = random.randint(0, 10000)
    print(f"Training ID: {training_id}")

    config_ = config()
    queue = mp.Queue(maxsize=config_.queue_size)

    policy_net = DQNModel(config_.obs_shape, config_.num_actions)
    policy_net.share_memory()

    processes = []

    for idx_env in range(config_.num_worker_env):
        env = mp.Process(target=Env, args=(idx_env, training_id, queue, policy_net))
        processes.append(env)
        env.start()

    for idx_trainer in range(config_.num_worker_trainer):
        trainer = mp.Process(target=Trainer, args=(idx_trainer, training_id, queue, policy_net))
        processes.append(trainer)
        trainer.start()

    for p in processes:
        p.join()
    

if __name__ == "__main__":
    main()
