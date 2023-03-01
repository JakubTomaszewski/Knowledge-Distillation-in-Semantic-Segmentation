import torch
from collections import defaultdict
from typing import List


class Evaluator:
    """Evaluator class for calculating the Intersection Over Union metric (Jaccard Index)
    """
    def __init__(self, class_labels: List) -> None:
        """
        Args:
            class_labels (List): list containing all possible labels
        
        Initializes:
            class_labels (List): list containing all possible labels
            total_iou_per_class (dict): dict with class labels as keys and the total IoU scores as values
            class_appearances (dict): dict with class labels as keys and number of their appearances as values

        """
        self.class_labels = class_labels
        self.total_iou_per_class = defaultdict(float)
        self.class_appearances = defaultdict(int)

    def __call__(self) -> dict:
        """Calls the mean_iou() function

        Returns:
            float: mean IoU value accross all classes
        """
        return self.mean_iou()

    def update_state(self, predictions: torch.Tensor, labels: torch.Tensor) -> None:
        """Updates the Evaluator class internal state by calculating the Intersection Over Union (IoU) metric for each class
        and adding it to the total_iou_per_class dict for the corresponding class.
        The updates are performed based on the predictions and labels supplied.
        
        Args:
            predictions (torch.Tensor): list of all predictions
            labels (torch.Tensor): list of all ground truth labels

        Raises:
            ValueError: when the provided predictions and labels lengths do not match
        """
        if predictions.shape != labels.shape:
            raise ValueError('predictions and labels must have the same shape')

        for pred, label in zip(predictions, labels):
            iou_scores = self.iou_score_per_class(pred, label, self.class_labels)
            for class_id, iou_value in iou_scores.items():
                if iou_value is not None:
                    self.class_appearances[class_id] += 1
                    self.total_iou_per_class[class_id] += iou_value
                else:
                    continue

    def reset_state(self):
        """Resets the Evaluator class internal state by reinitializing: total_iou_per_class and class_appearances dicts
        """
        self.total_iou_per_class = defaultdict(float)
        self.class_appearances = defaultdict(int)

    def mean_iou(self) -> float:
        """Calculates the overall mean IoU value accross all classes.

        Returns:
            float: mean IoU value accross all classes
        """
        ious_per_class = self.mean_iou_per_class()

        if len(ious_per_class) == 0:
            print('No samples evaluated')
            return None
        else:
            return sum(ious_per_class.values()) / len(ious_per_class)

    def mean_iou_per_class(self) -> dict:
        """Calculates the mean IoU value for each class.

        Returns:
            dict: dict with mean IoU values for each class in the following formar { 'class_id': IoU value }
        """
        iou_per_class_dict = {}
        for (class_id, total_iou), num_appearances in zip(self.total_iou_per_class.items(), self.class_appearances.values()):
            iou_per_class_dict[class_id] = total_iou / num_appearances
        return iou_per_class_dict

    def iou_score_per_class(self, y_pred: torch.Tensor, y_true: torch.Tensor, class_labels: List) -> dict:
        """Calculates the IoU value for a single sample, for each class respectively.

        Args:
            y_pred (torch.Tensor): predicted mask
            y_true (torch.Tensor): ground truth label
            class_labels (List): list containing all possible labels

        Returns:
            dict: dictionary with class labels as keys and IoUs as values: { 'class_id': iou_score }
        """
        class_iou_scores = {}
        class_stats = self._get_prediction_stats(y_pred, y_true, class_labels)

        for label, (tp, fp, fn) in class_stats.items():
            class_iou_scores[label] = self._iou_score(tp, fp, fn)

        return class_iou_scores

    def _get_prediction_stats(self, y_pred: torch.Tensor, y_true: torch.Tensor, class_labels: List) -> dict:
        """Calculates statistics for a single segmentation prediction such as:
        - number of True Positives (TP)
        - number of False Positives (FP)
        - number of False Negatives (FN)
        
        Args:
            y_pred (torch.Tensor): predicted mask
            y_true (torch.Tensor): ground truth label
            class_labels (List): list containing all possible labels

        Returns:
            dict: dict containing the class labels and the corresponding statistics in the following format:
                { 'class_id': (TP, FP, FN) }
        """
        class_stats = {}

        for label in class_labels:
            label_mask = y_true == label
            pred_mask = y_pred == label

            tp = torch.sum(torch.bitwise_and(label_mask, pred_mask)).item()
            fp = torch.sum(torch.bitwise_and(label_mask, pred_mask) ^ label_mask).item()
            fn = torch.sum(torch.bitwise_and(label_mask, pred_mask) ^ pred_mask).item()

            class_stats[label] = (tp, fp, fn)
        return class_stats

    def _iou_score(self, tp: int, fp: int, fn: int) -> float:
        """Calculates the Intersection Over Union metric (Jaccard Index).

        Args:
            tp (int): number of True Positives (TP)
            fp (int): number of False Positives (FP)
            fn (int): number of False Negatives (FN)

        Returns:
            float: Intersection Over Union metric (Jaccard Index) value
        """
        if (tp + fp + fn) == 0: # Zero division (class not present)
            return None
        else:
            return tp / (tp + fp + fn)
