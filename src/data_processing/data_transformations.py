from typing import Tuple
import numpy as np
import torch
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



# TODO: Create argParser for transformation params


def create_data_transformation_pipeline(img_shape: Tuple,
                                        rotation_angle: int,
                                        padding: int,
                                        distortion_factor: float,
                                        crop_factor: float,
                                        sharpness_factor: float,
                                        fill_value: int=0
                                        ) -> Compose:
    return Compose(
        [
            Resize(img_shape),
            Grayscale(1),
            CenterCrop((np.array(img_shape) * crop_factor).astype(int)),
            RandomRotation(rotation_angle, fill=fill_value),
            RandomPerspective(distortion_factor, p=1, fill=fill_value),
            RandomAdjustSharpness(sharpness_factor, p=1),
            RandomAutocontrast(p=1),
            Pad(padding, fill=fill_value)
        ])


def create_label_transformation_pipeline(img_shape: Tuple,
                                         rotation_angle: int,
                                         padding: int,
                                         distortion_factor: float,
                                         crop_factor: float,
                                         sharpness_factor: float,
                                         fill_value: int=0
                                         ) -> Compose:
    return Compose(
        [
            Resize(img_shape),
            CenterCrop((np.array(img_shape) * crop_factor).astype(int)),
            RandomRotation(rotation_angle, fill=fill_value),
            RandomPerspective(distortion_factor, p=1, fill=fill_value),
            RandomAdjustSharpness(sharpness_factor, p=1),
            RandomAutocontrast(p=1),
            Pad(padding, fill=fill_value)
        ])
