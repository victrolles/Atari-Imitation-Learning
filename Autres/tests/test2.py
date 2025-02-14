def scale_reward(r, max=10):
    return r / max

# Example
scaled_reward = scale_reward(10)  # Example input reward
print(scaled_reward)
