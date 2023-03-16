import torch.nn as nn
from argparse import ArgumentParser
from typing import Callable
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


def create_prediction_postprocessing_pipeline(config: ArgumentParser) -> Callable:
    """Factory function which creates a prediction postprocessing pipeline
    used for extracting argmax values and resizing the prediction to its original shape.

    Args:
        config (argparse.ArgumentParser): data config

    Returns:
        Callable: prediction postprocessing pipeline
    """
    img_shape = (config.img_height, config.img_width)

    def prediction_postprocessing_pipeline(output_mask):
        pred_mask_resized = nn.functional.interpolate(output_mask,  # outputs.logits.detach().cpu()
                            size=img_shape,  # img.shape[2:], # (height, width)
                            mode='bilinear',
                            align_corners=False)
        y_pred = pred_mask_resized.argmax(dim=1)[0]
        return y_pred
    return prediction_postprocessing_pipeline
