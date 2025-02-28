# Atari-Imitation-Learning

## setup

We used python 1.12.8 for this project.

### create a virtual environment
```bash
python -m venv venv
```

### activate the virtual environment
```bash
venv/Scripts/activate
```

### install dependencies
```bash
pip install -r requirements.txt
```

### Start the training
```bash
python _dqn_on_gym.py
```

### Open Tensorboard
```bash
tensorboard --logdir=./tensorboard
```

https://github.com/Div99/IQ-Learn