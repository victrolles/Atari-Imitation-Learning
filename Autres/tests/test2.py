import numpy as np
import matplotlib.pyplot as plt

def linear_schedule(start_e, end_e, duration, t):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

# Paramètres
start_e = 1.0    # Valeur initiale (ex: exploration max)
end_e = 0.1      # Valeur finale (ex: exploration min)
duration = 1000  # Nombre d'itérations pour décroître

# Génération des valeurs sur 1500 étapes
timesteps = np.arange(1500)
values = [linear_schedule(start_e, end_e, duration, t) for t in timesteps]

# Visualisation
plt.figure(figsize=(8, 4))
plt.plot(timesteps, values, label="Exploration Rate")
plt.axvline(x=duration, color='r', linestyle='--', label="End of decay")
plt.xlabel("Timesteps")
plt.ylabel("Value")
plt.title("Linear Decay Schedule")
plt.legend()
plt.grid()
plt.show()
