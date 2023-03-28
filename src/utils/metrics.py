import numpy as np
from transformers import EvalPrediction
from collections import defaultdict
from typing import List


class Evaluator:
    """Evaluator class for calculating the Intersection Over Union metric (Jaccard Index)
    """
    def __init__(self, 
                 class_labels: List,
                 ignore_index: int=255,
                 label_to_class_name: dict={}
                 ) -> None:
        """
        Args:
            class_labels (List): list containing all possible labels
            ignore_index (int, optional): class label which should be ignored during evaluation. Defaults to 255.
            label_to_class_name (dict, optional): dict mapping labels to their corresponding class names
            
        Initializes:
            class_tp (dict): dict containing True Positives for each class
            class_fp (dict): dict containing False Positives for each class
            class_fn (dict): dict containing False Negatives for each class
        """
        self.class_labels = class_labels
        self.ignore_index = ignore_index
        self.label_to_class_name = label_to_class_name

        self.class_tp = defaultdict(int)
        self.class_fp = defaultdict(int)
        self.class_fn = defaultdict(int)

    def __call__(self, y_pred: List[np.ndarray], y_true: List[np.ndarray]) -> dict:
        """Calls the mean_iou_score() function.

        Args:
            y_pred (List[np.ndarray]): _description_
            y_true (List[np.ndarray]): _description_

        Returns:
            float: mean IoU value accross all classes
        """
        return self.mean_iou_score(y_pred, y_true)

    def compute_metrics(self, eval_pred: EvalPrediction) -> dict:
        """Computes the Mean IoU metric and the IoU per class metric for provided predictions and labels.
        Works with HuggingFace transformers API.
        
        Args:
            eval_pred (EvalPrediction): input predictions and labels

        Returns:
            dict: dictionary with stored metrics
        """
        y_true, y_pred = eval_pred

        metrics = {}
        metrics["mean_iou"] = self.mean_iou_score(y_pred, y_true)
        metrics["iou_per_class"] = self.iou_score_per_class(y_pred, y_true)
        return metrics

    def mean_iou_score(self, y_pred: List[np.ndarray], y_true: List[np.ndarray]) -> float:
        """Calculates the overall mean IoU value accross all classes for given samples.

        Returns:
            float: mean IoU value accross all classes
        """
        ious_per_class = self.iou_score_per_class(y_pred, y_true)

        if len(ious_per_class) == 0:
            print('No samples evaluated')
            return None
        else:
            class_ious = ious_per_class.values()
            present_class_ious = list(filter(lambda x: x is not None, class_ious))
            return round(sum(present_class_ious) / len(present_class_ious), 4)

    def iou_score_per_class(self, y_pred: List[np.ndarray], y_true: List[np.ndarray]) -> dict:
        """Calculates the IoU value for a single sample, for each class respectively.

        Args:
            y_pred (List[np.ndarray]): list of all predictions (each prediction is a 2D array)
            y_true (List[np.ndarray]): list of all ground truth labels (each label is a 2D array)

        Returns:
            dict: dictionary with class labels as keys and IoUs as values: { 'class_id': iou_score }
        """
        class_iou_scores = {}
        class_tp, class_fp, class_fn  = self._get_prediction_stats(y_pred, y_true)

        for label_id in self.class_labels:
            tp = class_tp[label_id]
            fp = class_fp[label_id]
            fn = class_fn[label_id]
            class_iou_scores[label_id] = self._iou_score(tp, fp, fn)
        return class_iou_scores

    def update_state(self, y_pred: List[np.ndarray], y_true: List[np.ndarray]) -> None:
        """Updates the Evaluator class internal state by calculating the Intersection Over Union (IoU) metric for each class
        and adding it to the total_iou_per_class dict for the corresponding class.
        The updates are performed based on the predictions and labels supplied.
        
        Args:
            y_pred (List[np.ndarray]): list of all predictions (each prediction is a 2D array)
            y_true (List[np.ndarray]): list of all ground truth labels (each label is a 2D array)

        Raises:
            ValueError: when the provided predictions and labels lengths do not match
        """
        if len(y_pred) != len(y_true):
            raise ValueError('Predictions and labels must have the same shape')

        class_tp, class_fp, class_fn  = self._get_prediction_stats(y_pred, y_true)
        self._add_dicts(self.class_tp, class_tp)
        self._add_dicts(self.class_fp, class_fp)
        self._add_dicts(self.class_fn, class_fn)

    def reset_state(self):
        """Resets the Evaluator class internal state by reinitializing dicts containing True Positives, False Positives and False Negatives for each class.
        """
        self.class_tp = defaultdict(int)
        self.class_fp = defaultdict(int)
        self.class_fn = defaultdict(int)

    def internal_state_mean_iou(self) -> float:
        """Calculates the overall mean IoU value accross all classes, based on the internal state, which is updated using the `update_state` method.
        This can be used in a DataLoader sequential loop.

        Returns:
            float: mean IoU value accross all classes
        """
        ious_per_class = self.internal_state_iou_score_per_class()

        if len(ious_per_class) == 0:
            print('No samples evaluated')
            return None
        else:
            class_ious = ious_per_class.values()
            present_class_ious = list(filter(lambda x: x is not None, class_ious))
            return round(sum(present_class_ious) / len(present_class_ious), 4)

    def internal_state_iou_score_per_class(self) -> dict:
        """Calculates the mean IoU value for each class, based on the internal state, which is updated using the `update_state` method.
        This can be used in a DataLoader sequential loop.

        Returns:
            dict: dict with mean IoU values for each class in the following format { 'class_id': IoU value }
        """
        class_iou_scores = {}
        
        for label_id in self.class_labels:
            tp = self.class_tp[label_id]
            fp = self.class_fp[label_id]
            fn = self.class_fn[label_id]
            class_iou_scores[label_id] = self._iou_score(tp, fp, fn)
        return class_iou_scores

    def _iou_score(self, tp: int, fp: int, fn: int) -> float:
        """Calculates the Intersection Over Union metric (Jaccard Index).

        Args:
            tp (int): number of True Positives (TP)
            fp (int): number of False Positives (FP)
            fn (int): number of False Negatives (FN)

        Returns:
            float: Intersection Over Union metric (Jaccard Index) value
        """
        if (tp + fp + fn) == 0: # Zero division (class not present and not predicted)
            return None
        else:
            return tp / (tp + fp + fn)

    def _get_prediction_stats(self, y_pred: List[np.ndarray], y_true: List[np.ndarray]) -> dict:
        """Calculates statistics for a single segmentation prediction such as:
        - number of True Positives (TP)
        - number of False Positives (FP)
        - number of False Negatives (FN)
        
        Args:
            y_pred (List[np.ndarray]): list of all predictions (each prediction is a 2D array)
            y_true (List[np.ndarray]): list of all ground truth labels (each label is a 2D array)

        Returns:
            dict: dict containing the class labels and the corresponding statistics in the following format:
                { 'class_id': (TP, FP, FN) }
        """
        class_tp = defaultdict(int)
        class_fp = defaultdict(int)
        class_fn = defaultdict(int)

        for prediction, label in zip(y_pred, y_true):
            for label_id in self.class_labels:
                mask = label != self.ignore_index
                mask = np.not_equal(label, self.ignore_index)
                prediction = prediction[mask]
                label = np.array(label)[mask]

                label_mask = label == label_id
                pred_mask = prediction == label_id

                tp = np.sum(np.bitwise_and(label_mask, pred_mask))
                fp = np.sum(np.bitwise_and(label_mask, pred_mask) ^ label_mask)
                fn = np.sum(np.bitwise_and(label_mask, pred_mask) ^ pred_mask)

                if (tp + fp + fn) == 0: # If class not present and not predicted
                    continue

                class_tp[label_id] += tp
                class_fp[label_id] += fp
                class_fn[label_id] += fn
        return class_tp, class_fp, class_fn

    def _add_dicts(self, dict1: dict, dict2: dict) -> None:
        """Adds two dicts together inplace.
        
        Args:
            dict1 (dict): first dict
            dict2 (dict): second dict
        """
        for key, value in dict2.items():
            if key in dict1:
                dict1[key] = dict1[key] + value
            else:
                dict1[key] = value
