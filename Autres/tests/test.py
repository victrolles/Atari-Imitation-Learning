import math
def compute_output_size(H):
    # Première couche : Conv2d(1, 32, 3, stride=4)
    H1 = math.floor((H - 3) / 4) + 1
    
    # Deuxième couche : Conv2d(32, 64, 3, stride=3)
    H2 = math.floor((H1 - 3) / 3) + 1
    
    # Troisième couche : Conv2d(64, 64, 3, stride=1)
    H3 = math.floor((H2 - 3) / 1) + 1
    
    # Nombre total de valeurs après Flatten()
    output_size = H3 * H3 * 64
    
    return output_size

# Test avec différentes tailles d'entrée
sizes = [256, 128, 112, 100, 84, 83, 82, 81, 80, 65, 64]
results = {size: compute_output_size(size) for size in sizes}

# Affichage des résultats
for size, output in results.items():
    print(f"Input size: {size} -> Output size: {output}")