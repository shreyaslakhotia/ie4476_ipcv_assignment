"""Stateless preprocessing; no population statistics or fitted parameters."""

import numpy as np


def preprocess_images(images):
    """Scale uint8 grayscale images by 255 and flatten each in row-major order.

    Accepts (N,H,W), returns float32 (N,H*W). Apply this same function to future
    training/test subsets independently. Fit centering and dimensionality
    reduction on training features only in subsequent phases.
    """
    images = np.asarray(images)
    if images.ndim != 3 or images.dtype != np.uint8 or 0 in images.shape:
        raise ValueError("Expected nonempty uint8 grayscale images of shape (N,H,W)")
    return images.reshape(images.shape[0], -1).astype(np.float32) / np.float32(255)
