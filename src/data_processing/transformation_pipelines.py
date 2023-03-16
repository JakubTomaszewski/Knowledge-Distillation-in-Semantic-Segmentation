import sys
from argparse import ArgumentParser
import numpy as np
import torchvision
from torchvision.transforms import Compose
from torchvision.transforms import (PILToTensor,
                                    Grayscale,
                                    RandomRotation,
                                    CenterCrop,
                                    Pad,
                                    Resize,
                                    RandomPerspective,
                                    RandomAdjustSharpness,
                                    RandomAutocontrast,
                                    InterpolationMode
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
            # CenterCrop((np.array(img_shape) * config.crop_factor).astype(int)),
            # RandomRotation(config.max_rotation_angle, fill=config.void_class_id),
            # RandomPerspective(config.distortion_factor, p=1, fill=config.void_class_id),
            Pad(config.padding, fill=config.void_class_id)
        ]

    pixel_value_transformers = [
        RandomAutocontrast(p=0.5),
        RandomAdjustSharpness(config.sharpness_factor, p=0.5),
    ]

    return Compose(
        [
            PILToTensor(),
            # Grayscale(),
            # RandomTransformer(pixel_value_transformers, num_choices=1),
            RandomTransformer(aug_transformers, num_choices=config.num_transforms),
            # Resize(img_shape)
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
        # CenterCrop((np.array(img_shape) * config.crop_factor).astype(int)),
        # RandomRotation(config.max_rotation_angle, fill=config.void_class_id),
        # RandomPerspective(config.distortion_factor, p=1, fill=config.void_class_id, interpolation=InterpolationMode.NEAREST),
        Pad(config.padding, fill=config.void_class_id)
    ]

    return Compose(
        [
            PILToTensor(),
            RandomTransformer(aug_transformers, num_choices=config.num_transforms),
            # Resize(img_shape, interpolation=InterpolationMode.NEAREST)
        ])


def create_evaluation_data_transformation_pipeline(config: ArgumentParser,
                                        ) -> torchvision.transforms.Compose:
    """Generates a data transformation pipeline for evaluation.

    Args:
        config (argparse.ArgumentParser): data transformation config

    Returns:
        torchvision.transforms.Compose: image transformation pipeline
    """
    img_shape = (config.img_height, config.img_width)

    return Compose(
        [
            PILToTensor(),
            # Grayscale(),
            # Resize(img_shape)
        ])

def create_evaluation_label_transformation_pipeline(config: ArgumentParser,
                                         ) -> torchvision.transforms.Compose:
    """Generates a label transformation pipeline for evaluation.

    Args:
        config (argparse.ArgumentParser): label transformation config

    Returns:
        torchvision.transforms.Compose: image transformation pipeline
    """
    img_shape = (config.img_height, config.img_width)

    return Compose(
        [
            PILToTensor(),
            # Resize(img_shape, interpolation=InterpolationMode.NEAREST)
        ])

