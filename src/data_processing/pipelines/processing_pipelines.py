import torch
import torch.nn as nn
from argparse import ArgumentParser
from typing import Callable, Tuple
from transformers import SegformerImageProcessor


def create_data_preprocessing_pipeline(config: ArgumentParser) -> Callable:
    """Factory function which creates a data preprocessing pipeline used for resizing an input image.

    Args:
        config (argparse.ArgumentParser): data config

    Returns:
        Callable: image preprocessing pipeline
    """
    data_preprocessor = SegformerImageProcessor.from_pretrained(config.model_checkpoint)
    data_preprocessor.size = {'height': config.img_height, 'width': config.img_width}
    return data_preprocessor


def create_prediction_postprocessing_pipeline(img_shape: Tuple[int, int]) -> Callable:
    """Factory function which creates a prediction postprocessing pipeline
    used for extracting argmax values and resizing the prediction to its original shape.

    Args:
        img_shape (Tuple[int, int]): tuple containing the desired img_shape (height, width)

    Returns:
        Callable: prediction postprocessing pipeline
    """
    def prediction_postprocessing_pipeline(prediction: torch.Tensor, _):
        """Scales the output prediction to the desired size and extracts model's class predictions with the highiest confidence.

        Args:
            prediction (torch.Tensor): model predictions
            _ (torch.Tensor): ground truth labels. The param is not used, it is required to match the HuggingFace transformers library

        Returns:
            torch.Tensor: prediction after postprocessing
        """
        prediction_resized = nn.functional.interpolate(prediction,
                            size=img_shape, # (height, width)
                            mode='bilinear',
                            align_corners=False)
        y_pred = prediction_resized.argmax(dim=1)
        return y_pred
    return prediction_postprocessing_pipeline
