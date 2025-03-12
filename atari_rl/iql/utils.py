import random

import torch

# Fonction pour équilibrer un dataset d'Experience
def balance_experience_dataset(experiences: list) -> list:
    # Compter les occurrences de chaque action
    action_counts = {}
    
    for exp in experiences:
        if exp.action not in action_counts:
            action_counts[exp.action] = 0
        action_counts[exp.action] += 1
    
    # Trouver l'action avec le moins d'occurrences
    min_count = min(action_counts.values())
    
    # Créer un nouveau dataset équilibré
    balanced_experiences = []
    
    # Pour chaque action, échantillonner au hasard pour ne garder que min_count éléments
    for action in action_counts:
        # Filtrer les éléments de cette action
        action_items = [exp for exp in experiences if exp.action == action]
        
        # Si la classe a plus d'éléments que le minimum, échantillonner
        if len(action_items) > min_count:
            action_items = random.sample(action_items, min_count)
        
        # Ajouter les éléments échantillonnés ou tous les éléments si déjà équilibrés
        balanced_experiences.extend(action_items)
    
    return balanced_experiences

def get_concat_samples(policy_batch: dict, expert_batch: dict, device: torch.device):
    """Concatenate policy and expert samples into a single batch."""

    # Concatenate the samples
    batch_state = torch.cat([policy_batch['states'], expert_batch['states']], dim=0)
    batch_next_state = torch.cat([policy_batch['next_states'], expert_batch['next_states']], dim=0)
    batch_action = torch.cat([policy_batch['actions'].squeeze(1), expert_batch['actions']], dim=0)
    batch_reward = torch.cat([policy_batch['rewards'].squeeze(1), torch.zeros_like(expert_batch['dones'], dtype=torch.float32, device=device)], dim=0)
    batch_done = torch.cat([policy_batch['dones'].squeeze(1), expert_batch['dones']], dim=0)
    is_expert = torch.cat([torch.zeros_like(expert_batch['dones'], dtype=torch.bool, device=device),
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