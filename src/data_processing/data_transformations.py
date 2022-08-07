import numpy as np
from typing import Tuple
import torch
from torchvision.transforms import Grayscale, RandomRotation, RandomCrop, Pad, Resize, RandomPerspective
from torchvision.transforms import Compose


# TODO: Create argParser for transformation params


def create_data_transformation_pipeline(img_shape: Tuple, rotation_angle, padding, distortion_factor, fill_value=0) -> Compose:
    return Compose([Resize(img_shape),
                    Grayscale(1),
                    RandomRotation(rotation_angle, fill=fill_value),
                    Pad(padding, fill=fill_value),
                    RandomPerspective(distortion_factor, p=1, fill=fill_value)
                    # RandomCrop(0.9)
                    ])


def create_label_transformation_pipeline(img_shape: Tuple, rotation_angle, padding, distortion_factor, fill_value=0) -> Compose:
    return Compose([Resize(img_shape),
                    RandomRotation(rotation_angle, fill=0),
                    Pad(padding, fill=fill_value),
                    RandomPerspective(distortion_factor, p=1, fill=fill_value)
                    ])
