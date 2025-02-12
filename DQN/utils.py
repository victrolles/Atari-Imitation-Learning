import cv2
import numpy as np
import math

def prepost_image_state(state, image_size = 128, game_name = "MsPacman-v5"):
    # Crop the image
    if game_name == "MsPacman-v5":
        cropped_state = state[1:171, 0:159]
    elif game_name == "Enduro-v5":
        cropped_state = state[0:154, 8:158]
    else:
        raise ValueError("Invalid game name")

    # Resize the image to 128x128
    resized_state = cv2.resize(cropped_state, (image_size, image_size))

    # Convert to grayscale
    gray_state = cv2.cvtColor(resized_state, cv2.COLOR_BGR2GRAY)

    # Normalize the image
    normalized_state = gray_state.astype(np.float32) / 255.0

    # Add one dimension to the image
    one_channel_state = np.expand_dims(normalized_state, axis=0)

    return one_channel_state

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