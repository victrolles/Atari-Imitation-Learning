import torch
import torch.nn as nn
import torch.optim as optim

# Définir un réseau de neurones simple pour le Q-learning
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fc(x)

# Classe principale avec la méthode getV
class QLearningAgent:
    def __init__(self, input_dim, output_dim, alpha=1.0):
        self.q_net = QNetwork(input_dim, output_dim)  # Réseau pour Q-learning
        self.alpha = alpha  # Paramètre alpha
    
    def getV(self, obs):
        # Calcul de la valeur (v) avec logsumexp
        q = self.q_net(obs)
        v = self.alpha * torch.logsumexp(q / self.alpha, dim=1, keepdim=True)
        return v

# Fonction de test
def test_getV():
    # Paramètres pour le test
    input_dim = 4  # Exemple : entrée de dimension 4 (par exemple, 4 observations)
    output_dim = 3  # Exemple : 3 actions possibles
    alpha = 0.5  # Valeur de alpha
    
    # Créer un agent QLearning
    agent = QLearningAgent(input_dim, output_dim, alpha)
    
    # Créer un batch d'observations factices
    batch_size = 5
    obs = torch.randn(batch_size, input_dim)  # Batch de 5 observations avec 4 features par observation
    
    # Calculer la valeur (v) pour ces observations
    v = agent.getV(obs)
    
    # Afficher les résultats
    print("Observations:")
    print(obs)
    print("\nValeur (v) calculée:")
    print(v)

# Exécuter le test
test_getV()
