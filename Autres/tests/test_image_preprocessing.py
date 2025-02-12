import gymnasium as gym
import cv2
import numpy as np

# Initialize the environment
import ale_py
game_name = "MsPacman-v5" # "MsPacman-v5" or "Enduro-v5"
gym.register_envs(ale_py) 
env = gym.make(f"ALE/{game_name}", render_mode="rgb_array")

# create a folder to save the images
import os
os.makedirs(f"Autres/images/{game_name}", exist_ok=True)

# Reset the environment and get the first frame
state, _ = env.reset()

print(state.shape)  # (210, 160, 3)

# Convert RGB to BGR (OpenCV uses BGR format)
state_bgr = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)
cv2.imwrite(f"Autres/images/{game_name}/frame.png", state_bgr)

# Crop the image
cropped_frame = state_bgr[1:171, 0:159] # MsPacman
# cropped_frame = state_bgr[0:154, 8:158] # Enduro
cv2.imwrite(f"Autres/images/{game_name}/cropped_frame.png", cropped_frame)

# Resize the image to 128x128
resized_frame = cv2.resize(cropped_frame, (84, 84))
cv2.imwrite(f"Autres/images/{game_name}/resized_frame.png", resized_frame)

# Convert to grayscale
gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
cv2.imwrite(f"Autres/images/{game_name}/gray_frame.png", gray_frame)

# Normalize the image
normalized_frame = gray_frame.astype(np.float32) / 255.0
print(normalized_frame)

# Add one dimension to the image
final_state = np.expand_dims(normalized_frame, axis=0)
print(final_state.shape)

env.close()
