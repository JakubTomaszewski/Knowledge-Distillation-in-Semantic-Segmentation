from typing import Tuple
import numpy as np
import torchvision
from torchvision.transforms import Compose
from torchvision.transforms import (
                                    Grayscale,
                                    RandomRotation,
                                    CenterCrop,
                                    Pad,
                                    Resize,
                                    RandomPerspective,
                                    RandomAdjustSharpness,
                                    RandomAutocontrast
                                    )

from data_processing.transformers.random_transform_choice import RandomTransformChoice


# TODO: Create argParser for transformation params


def create_data_transformation_pipeline(img_shape: Tuple,
                                        rotation_angle: int,
                                        padding: int,
                                        distortion_factor: float,
                                        crop_factor: float,
                                        sharpness_factor: float,
                                        fill_value: int=0
                                        ) -> torchvision.transforms.Compose:
    """Generates a training data transformation pipeline.

    Args:
        img_shape (Tuple): _description_
        rotation_angle (int): _description_
        padding (int): _description_
        distortion_factor (float): _description_
        crop_factor (float): _description_
        sharpness_factor (float): _description_
        fill_value (int, optional): _description_. Defaults to 0.

    Returns:
        torchvision.transforms.Compose: _description_
    """

    aug_transformers = [
            CenterCrop((np.array(img_shape) * crop_factor).astype(int)),
            RandomRotation(rotation_angle, fill=fill_value),
            RandomPerspective(distortion_factor, p=1, fill=fill_value),
            RandomAdjustSharpness(sharpness_factor, p=1),
            RandomAutocontrast(p=1),
            Pad(padding, fill=fill_value)
        ]

    random_picker = RandomTransformChoice(aug_transformers, num_choices=3)
    random_transformers = random_picker()

    return Compose(
        [
            Resize(img_shape),
            Grayscale(1),
            *random_transformers
        ])


def create_label_transformation_pipeline(img_shape: Tuple,
                                         rotation_angle: int,
                                         padding: int,
                                         distortion_factor: float,
                                         crop_factor: float,
                                         sharpness_factor: float,
                                         fill_value: int=0
                                         ) -> torchvision.transforms.Compose:
    """Generates a training label transformation pipeline.

    Args:
        img_shape (Tuple): _description_
        rotation_angle (int): _description_
        padding (int): _description_
        distortion_factor (float): _description_
        crop_factor (float): _description_
        sharpness_factor (float): _description_
        fill_value (int, optional): _description_. Defaults to 0.

    Returns:
        torchvision.transforms.Compose: _description_
    """

    aug_transformers = [
            CenterCrop((np.array(img_shape) * crop_factor).astype(int)),
            RandomRotation(rotation_angle, fill=fill_value),
            RandomPerspective(distortion_factor, p=1, fill=fill_value),
            RandomAdjustSharpness(sharpness_factor, p=1),
            RandomAutocontrast(p=1),
            Pad(padding, fill=fill_value)
        ]

    random_picker = RandomTransformChoice(aug_transformers, num_choices=3)
    random_transformers = random_picker()

    return Compose(
        [
            Resize(img_shape),
            *random_transformers
        ])
