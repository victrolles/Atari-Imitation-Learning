import torch

def get_concat_samples(policy_batch: dict, expert_batch: dict, device: torch.device):
    """Concatenate policy and expert samples into a single batch."""
    batch_state = torch.cat([policy_batch['states'], expert_batch['states']], dim=0)
    batch_next_state = torch.cat([policy_batch['next_states'], expert_batch['next_states']], dim=0)
    batch_action = torch.cat([policy_batch['actions'], expert_batch['actions']], dim=0)
    batch_reward = torch.cat([policy_batch['rewards'], torch.zeros_like(expert_batch['dones'], dtype=torch.float32, device=device)], dim=0)
    batch_done = torch.cat([policy_batch['dones'], expert_batch['dones']], dim=0)
    is_expert = torch.cat([torch.zeros_like(policy_batch['dones'], dtype=torch.bool, device=device),
                            torch.ones_like(expert_batch['dones'], dtype=torch.bool, device=device)], dim=0)
    
    # Shuffle the batch
    indices = torch.randperm(batch_state.shape[0])
    batch_state = batch_state[indices]
    batch_next_state = batch_next_state[indices]
    batch_action = batch_action[indices]
    batch_reward = batch_reward[indices]
    batch_done = batch_done[indices]
    is_expert = is_expert[indices]

    return batch_state, batch_next_state, batch_action, batch_reward, batch_done, is_expert

# Example of use and verification
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_batch = {
        "states": torch.rand(10, 4, 4).to(device),
        "next_states": torch.rand(10, 4, 4).to(device),
        "actions": torch.randint(0, 4, (10,)).to(device),
        "rewards": torch.rand(10, 1).to(device),
        "dones": torch.rand(10, 1).to(device)
    }

    expert_batch = {
        "states": torch.rand(10, 4, 4).to(device),
        "next_states": torch.rand(10, 4, 4).to(device),
        "actions": torch.randint(0, 4, (10,)).to(device),
        "dones": torch.rand(10, 1).to(device)
    }

    print(expert_batch['states'].shape[0])

    batch_state, batch_next_state, batch_action, batch_reward, batch_done, is_expert = get_concat_samples(policy_batch, expert_batch, device)

    print(f"batch_state: {batch_state.shape}")
    print(f"batch_next_state: {batch_next_state.shape}")
    print(f"batch_action: {batch_action.shape}")
    print(f"batch_reward: {batch_reward.shape}")
    print(f"batch_done: {batch_done.shape}")
    print(f"is_expert: {is_expert.shape}")