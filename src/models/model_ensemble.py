import torch
import torch.nn as nn
from typing import Iterable, Callable



class SemanticSegmentationModelEnsemble(nn.Module):
    """Class representing an ensemble of models for semantic segmentation"""
    def __init__(self, models: Iterable[nn.Module], prediction_postprocessing_pipeline: Callable=None):
        """Initializes the model ensemble

        Args:
            models (Iterable[nn.Module]): an iterable of models to be used in the ensemble
            prediction_postprocessing_pipeline (Callable): a pipeline for postprocessing the predictions of the models
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.prediction_postprocessing_pipeline = prediction_postprocessing_pipeline

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model ensemble

        Args:
            X (torch.Tensor): input sample

        Returns:
            torch.Tensor: prediction of the model ensemble
        """
        predictions = [model(X).logits for model in self.models]
        
        if self.prediction_postprocessing_pipeline is not None:
            predictions = [self.prediction_postprocessing_pipeline(prediction) for prediction in predictions]
        
        predictions = torch.stack(predictions, dim=1)
        predictions = self.class_choice_strategy(predictions)
        return predictions

    def class_choice_strategy(self, predictions: torch.Tensor) -> torch.Tensor:
        """The strategy for choosing the class from the predictions of the model ensemble.
        By default, the most frequent class is chosen. Can be overridden in subclasses.

        Args:
            predictions (torch.Tensor): predictions of the models

        Returns:
            torch.Tensor: output with a single class for each pixel value
        """
        return torch.mode(predictions, dim=1).values
