import os
import random

import torch
import numpy as np

def select_action(
    policy_net,
    device: torch.device,
    num_actions: int,
    state: np.ndarray,
    epsilon: float = 0.0,
    training: bool = True,
    deterministic: bool = True,
    temperature: float = 1) -> int:
    
    """
    Select an action using epsilon-greedy strategy

    :param state: state of the environment
    :param epsilon: exploration rate (0.0 -> 1.0)
    :param training: whether the agent is training or not
    :param deterministic: whether to select the action deterministically or not
    :return: action index
    """
    policy_net.to(device)

    if training and random.random() < epsilon:
        return random.randrange(num_actions)
    else:
        with torch.no_grad():
            state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
            output = policy_net(state)

            if deterministic:
                output = torch.argmax(output, dim=1)
            else:
                output = torch.multinomial(torch.softmax(output / temperature, dim=1), 1)

            return int(output.cpu().item())

        
def save_model(
    policy_net: torch.nn.Module,
    model_path: str = None,
    model_name: str = None):
    
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        model_full_path = f"{model_path}/{model_name}.pt"
        print(f"Model saved at {model_full_path}")

        torch.save(policy_net.state_dict(), model_full_path)
        

def load_model(
    policy_net: torch.nn.Module,
    model_path: str = None,
    model_name: str = None):
        
    model_full_path = f"{model_path}/{model_name}.pt"
    print(f"Model loaded from {model_full_path}")

    return policy_net.load_state_dict(torch.load(model_full_path, weights_only=True))