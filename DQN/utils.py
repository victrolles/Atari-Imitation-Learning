import cv2
import numpy as np

def prepost_image_state(state):
    # Crop the image
    cropped_state = state[0:154, 8:158]

    # Resize the image to 128x128
    resized_state = cv2.resize(cropped_state, (128, 128))

    # Convert to grayscale
    gray_state = cv2.cvtColor(resized_state, cv2.COLOR_BGR2GRAY)

    # Normalize the image
    normalized_state = gray_state.astype(np.float32) / 255.0

    # Add one dimension to the image
    one_channel_state = np.expand_dims(normalized_state, axis=0)

    return one_channel_state