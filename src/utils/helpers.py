"""Helper functions
"""

import random
import torch
import numpy as np


def torch_image_to_numpy(image: torch.Tensor) -> np.ndarray:
    """Converts a torch.Tensor image to a numpy array.

    Args:
        image (torch.Tensor): image to convert

    Returns:
        np.ndarray: converted image
    """
    return image.permute(1, 2, 0).numpy()


def set_randomness_seed(seed):
    """Sets a seed for computations performed by torch and random library.

    Args:
        seed (int): random seed to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
