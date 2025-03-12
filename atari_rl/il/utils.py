import random

def balance_dataset(state_actions: list) -> list:
    # Compter les occurrences de chaque action
    action_counts = {}
    
    for sa in state_actions:
        if sa.action not in action_counts:
            action_counts[sa.action] = 0
        action_counts[sa.action] += 1
    
    # Trouver l'action avec le moins d'occurrences
    min_count = min(action_counts.values())
    
    # Créer un nouveau dataset équilibré
    balanced_state_actions = []
    
    # Pour chaque action, échantillonner au hasard pour ne garder que min_count éléments
    for action in action_counts:
        # Filtrer les éléments de cette action
        action_items = [sa for sa in state_actions if sa.action == action]
        
        # Si la classe a plus d'éléments que le minimum, échantillonner
        if len(action_items) > min_count:
            action_items = random.sample(action_items, min_count)
        
        # Ajouter les éléments échantillonnés ou tous les éléments si déjà équilibrés
        balanced_state_actions.extend(action_items)
    
    return balanced_state_actions