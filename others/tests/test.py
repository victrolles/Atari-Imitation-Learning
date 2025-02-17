import torch

x = torch.randn(1, 3, 1, 5)  # Shape: (1, 3, 1, 5)
print(x.shape)

y = x.squeeze(-1)  # Removes all dimensions of size 1
print(y.shape)   # Output: torch.Size([3, 5])

y = x.squeeze(0)  # Removes all dimensions of size 1
print(y.shape)   # Output: torch.Size([3, 5])

y = x.squeeze(1)  # Removes all dimensions of size 1
print(y.shape)   # Output: torch.Size([3, 5])

y = x.squeeze(2)  # Removes all dimensions of size 1
print(y.shape)   # Output: torch.Size([3, 5])

y = x.squeeze()  # Removes all dimensions of size 1
print(y.shape)   # Output: torch.Size([3, 5])

y = x.unsqueeze(1)  # Removes all dimensions of size 1
print(y.shape)   # Output: torch.Size([3, 5])
