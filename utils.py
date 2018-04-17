import numpy as np


def normalize_image(image):
    """Convert pixel intensity values from [0, 255] to [0.0, 1.0]."""
    return np.multiply(image.astype(np.float32), 1.0 / 255.0)