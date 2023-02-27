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


def get_device():
    """Returns the available device for computation.

    Returns:
        torch.device: available device for computation
    """
    compute_device = None
    if torch.cuda.is_available():
        compute_device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        compute_device = torch.device('mps')
    else:
        compute_device = torch.device('cpu')
    return compute_device


def display_dict(dict_to_print):
    """Displays a dict in a fancy way

    Args:
        dict_to_print (_type_): _description_
    """
    print("\n".join("{}:  {}".format(k, v) for k, v in sorted(dict_to_print.items())))