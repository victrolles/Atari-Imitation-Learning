import torch

# Un tensor source de taille (3, 4)
tensor = torch.tensor([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12]])

# Un tensor d'indices qui nous dit quelles valeurs récupérer
# Imaginons que l'on veuille récupérer les éléments dans la dimension 1 (colonnes)
index = torch.tensor([0, 1, 3]) # Ligne 2, indices de colonnes

# On utilise gather pour récupérer les valeurs spécifiées par les indices
result = tensor.gather(0, index)

print(result)
