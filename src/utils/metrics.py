import torch
from collections import defaultdict
from typing import List


def iou_score_per_class(y_pred: torch.Tensor, y_true: torch.Tensor, class_labels: List) -> dict:
    """Calculates the Intersection Over Union metric (Jaccard Index) for a single sample.
    The metric is calculated for each class respectively.

    Args:
        y_pred (torch.Tensor): predicted mask
        y_true (torch.Tensor): ground truth label
        class_labels (List): list containing all possible labels

    Returns:
        dict: dictionary with class labels as keys and IoUs as values: { 'class_id': iou_score }
    """
    class_iou_scores = {}
    class_stats = get_prediction_stats(y_pred, y_true, class_labels)

    for label, (tp, fp, fn) in class_stats.items():
        class_iou_scores[label] = iou_score(tp, fp, fn)

    return class_iou_scores


def iou_score(tp: int, fp: int, fn: int) -> float:
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


def get_prediction_stats(y_pred: torch.Tensor, y_true: torch.Tensor, class_labels: List) -> dict:
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


def _mean_iou_per_class(total_iou_per_class: dict, class_appearances: dict) -> dict:
    """Calculates the mean IoU for each class based on the total value and the number of class appearances.

    Args:
        total_iou_per_class (dict): dict with class labels as keys and the total IoU scores as values
        class_appearances (dict): dict with class labels as keys and number of their appearances as values

    Returns:
        dict: _description_
    """
    mean_iou_per_class = {}
    for (class_id, total_iou), num_appearances in zip(total_iou_per_class.items(), class_appearances.values()):
        mean_iou_per_class[class_id] = total_iou / num_appearances
    return mean_iou_per_class


def evaluate_predictions(predictions: torch.Tensor, labels: torch.Tensor, class_labels: List) -> dict:
    """Evaluates predictions by calculating the mean Intersection Over Union (IoU) for each class.

    Args:
        predictions (torch.Tensor): list of all predictions
        labels (torch.Tensor): list of all ground truth labels
        class_labels (List): list containing all possible labels

    Raises:
        ValueError: when the provided predictions and labels lengths do not match

    Returns:
        dict: dict with mean IoU values for each class
    """
    total_iou_per_class = defaultdict(float)
    class_appearances = defaultdict(int)

    if predictions.shape != labels.shape:
        raise ValueError('predictions and labels must have the same shape')

    for pred, label in zip(predictions, labels):
        iou_scores = iou_score_per_class(pred, label, class_labels)
        for class_id, iou_value in iou_scores.items():
            if iou_value is not None:
                class_appearances[class_id] += 1
                total_iou_per_class[class_id] += iou_value
            else:
                continue

    mean_iou_per_class = _mean_iou_per_class(total_iou_per_class, class_appearances)
    return mean_iou_per_class
