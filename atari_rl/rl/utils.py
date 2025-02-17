import cv2
import numpy as np
import math

def prepost_frame(frame, resize_size = 128, game_name = "MsPacman-v5"):
    """Prepost a frame.

    Apply the following steps to a frame:
    1. Crop the frame.
    2. Resize the frame.
    3. Convert the frame to grayscale.
    4. Normalize the frame.

    Args:
        frame (np.ndarray): The frame to prepost. Shape: (H, W, 3)
        resize_size (int): The size of the image.
        game_name (str): The name of the game.

    Returns:
        np.ndarray: The preposted frame. Shape: (resize_size, resize_size)
    """
    # Crop the image
    if game_name == "MsPacman-v5":
        cropped_frame = frame[1:171, 0:159]
    elif game_name == "Enduro-v5":
        cropped_frame = frame[0:154, 8:158]
    else:
        raise ValueError("Invalid game name")

    # Resize the image to 128x128
    resized_frame = cv2.resize(cropped_frame, (resize_size, resize_size))

    # Convert to grayscale
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # Normalize the image
    normalized_frame = gray_frame.astype(np.float32) / 255.0

    return normalized_frame

def compute_output_size(H):
    # Première couche : Conv2d(1, 32, 8, stride=4)
    H1 = math.floor((H - 8) / 4) + 1
    
    # Deuxième couche : Conv2d(32, 64, 4, stride=3)
    H2 = math.floor((H1 - 4) / 3) + 1
    
    # Troisième couche : Conv2d(64, 64, 3, stride=1)
    H3 = math.floor((H2 - 3) / 1) + 1
    
    # Nombre total de valeurs après Flatten()
    output_size = H3 * H3 * 64
    
    return output_size

def scale_reward(r, max=10):
    return r / max