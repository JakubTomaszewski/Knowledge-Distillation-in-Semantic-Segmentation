from argparse import ArgumentParser
import torchvision
from torchvision.transforms import (Compose,
                                    PILToTensor,
                                    Resize,
                                    RandomCrop,
                                    RandomHorizontalFlip,
                                    InterpolationMode
                                    )



def create_data_transformation_pipeline(config: ArgumentParser,
                                        ) -> torchvision.transforms.Compose:
    """Generates a training data transformation pipeline.

    Args:
        config (argparse.ArgumentParser): data transformation config

    Returns:
        torchvision.transforms.Compose: image transformation pipeline
    """
    img_shape = (config.img_height, config.img_width)

    return Compose(
        [
            PILToTensor(),
            Resize(img_shape, interpolation=InterpolationMode.BILINEAR),
            RandomCrop(config.crop_size),
            RandomHorizontalFlip(config.horizontal_flip_probability),
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

    return Compose(
        [
            PILToTensor(),
            Resize(img_shape, interpolation=InterpolationMode.NEAREST),
            RandomCrop(config.crop_size),
            RandomHorizontalFlip(config.horizontal_flip_probability),
        ])


def create_evaluation_data_transformation_pipeline(config: ArgumentParser,
                                        ) -> torchvision.transforms.Compose:
    """Generates a data transformation pipeline for evaluation.

    Args:
        config (argparse.ArgumentParser): data transformation config

    Returns:
        torchvision.transforms.Compose: image transformation pipeline
    """
    return Compose(
        [
            PILToTensor(),
        ])

def create_evaluation_label_transformation_pipeline(config: ArgumentParser,
                                         ) -> torchvision.transforms.Compose:
    """Generates a label transformation pipeline for evaluation.

    Args:
        config (argparse.ArgumentParser): label transformation config

    Returns:
        torchvision.transforms.Compose: image transformation pipeline
    """
    return Compose(
        [
            PILToTensor(),
        ])
