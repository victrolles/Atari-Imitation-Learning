class config():
    queue_size = 30000

    num_worker_env = 1
    num_worker_trainer = 2

    game_name = "MsPacman-v5"
    rl_algorithm = "DQN"
    num_actions = 5

    image_size = 84
    frame_stack_size = 3
    frame_skip_size = 4
    obs_shape = (frame_stack_size, image_size, image_size)

    gamma = 0.99
    learning_rate = 1e-4
    buffer_size = 20000
    batch_size = 32
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.99996
    target_update = 3
    max_step_per_episode = 10000
    iter_per_episode = 128

    num_episodes = 300000
    use_deterministic = True
    temperature = 1

    model_name = "DQN_MsPacman-v5_290_10700.pt"
    save_model = True
    load_model = False
    eval_rate = 200

    num_episodes_eval = 10
    epsilon_eval = 0.05
    use_deterministic_eval = False
    use_epsilon_eval = True
    temperature_eval = 1