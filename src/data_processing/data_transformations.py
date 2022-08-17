import sys
from argparse import ArgumentParser
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

sys.path.append('..')

from data_processing.transformers.random_transformer import RandomTransformer



def create_data_transformation_pipeline(config: ArgumentParser,
                                        ) -> torchvision.transforms.Compose:
    """Generates a training data transformation pipeline.

    Args:
        config (argparse.ArgumentParser): data transformation config

    Returns:
        torchvision.transforms.Compose: image transformation pipeline
    """
    img_shape = (config.img_height, config.img_width)

    aug_transformers = [
            CenterCrop((np.array(img_shape) * config.crop_factor).astype(int)),
            RandomRotation(config.max_rotation_angle, fill=config.fill_value),
            RandomPerspective(config.distortion_factor, p=1, fill=config.fill_value),
            RandomAdjustSharpness(config.sharpness_factor, p=1),
            RandomAutocontrast(p=1),
            Pad(config.padding, config.fill_value)
        ]

    return Compose(
        [
            Resize(img_shape),
            Grayscale(),
            RandomTransformer(aug_transformers, num_choices=config.num_transformers)
        ])


def create_label_transformation_pipeline(config: ArgumentParser,
                                         ) -> torchvision.transforms.Compose:
    """Generates a training label transformation pipeline.

    Args:
        config (argparse.ArgumentParser): label transformation config

    Returns:
        torchvision.transforms.Compose: image transformation pipeline
    """
    img_shape = (config.img_height, config.img_width)

    aug_transformers = [
        CenterCrop((np.array(img_shape) * config.crop_factor).astype(int)),
        RandomRotation(config.max_rotation_angle, fill=config.fill_value),
        RandomPerspective(config.distortion_factor, p=1, fill=config.fill_value),
        RandomAdjustSharpness(config.sharpness_factor, p=1),
        RandomAutocontrast(p=1),
        Pad(config.padding, config.fill_value)
    ]

    return Compose(
        [
            Resize(img_shape),
            RandomTransformer(aug_transformers, num_choices=config.num_transformers)
        ])
