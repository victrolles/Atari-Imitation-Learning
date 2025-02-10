import gymnasium as gym
import pygame
import numpy as np

# Initialiser Gym et Pygame
import ale_py
gym.register_envs(ale_py)
env = gym.make("ALE/Enduro-v5", render_mode="human")
_, _ = env.reset()
pygame.init()

screen = pygame.display.set_mode((400, 300))
clock = pygame.time.Clock()  # Limite la vitesse de la boucle

key_to_action = {
    pygame.K_LEFT: 3,  
    pygame.K_RIGHT: 2,  
    pygame.K_UP: 1,  
    pygame.K_DOWN: 4,  
}

running = True
while running:
    # Vider la file d'attente d'événements
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Vérifier les touches pressées
    keys = pygame.key.get_pressed()
    action = 0  # Ne rien faire par défaut

    if keys[pygame.K_LEFT]:
        action = 3
    elif keys[pygame.K_RIGHT]:
        action = 2
    elif keys[pygame.K_UP]:
        action = 1
    elif keys[pygame.K_DOWN]:
        action = 4

    # Exécuter une action seulement si nécessaire
    _, reward, _, _, _ = env.step(action)

    print(reward)

    clock.tick(30)  # Limite à 30 FPS pour éviter la surcharge

env.close()
pygame.quit()
