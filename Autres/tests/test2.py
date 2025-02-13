import torch

q_values = torch.tensor([
    [10.0, 20.0, 30.0],  # Q-values for state 1
    [5.0, 15.0, 25.0]    # Q-values for state 2
])  # Shape: (2, 3) -> 2 states, 3 actions each

actions = torch.tensor([[2], [1]])  # Chosen action indices (column-wise selection)

selected_q_values = q_values.gather(1, actions)  # Select Q-values for chosen actions

print(selected_q_values)
