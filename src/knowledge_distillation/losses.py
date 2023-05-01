import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import resize_outputs

### Response-based losses ###


class DistillationCrossEntropyLoss:
    def __init__(self,
                 temperature: int=1,
                 alpha: float=0.7,
                 ignore_index: int = -100
                 ) -> None:
        self.standard_targets_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.soft_targets_loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.temperature = temperature
        self.alpha = alpha
        self.ignore_index = ignore_index

    def __call__(self, student_logits, teacher_logits, target):
        # Resize teacher logits to match the size of student logits
        resized_teacher_logits = resize_outputs(teacher_logits, student_logits.shape[2:])
        # Resize student logits to match the target size
        resized_student_logits = resize_outputs(student_logits, target.shape[-2:])

        # Standard CrossEntropyLoss
        hard_labels_loss = self.standard_targets_loss(resized_student_logits, target)

        # Distillation loss
        soft_labels_loss = self.soft_targets_loss(self.log_softmax(student_logits / self.temperature),
                                                  self.softmax(resized_teacher_logits / self.temperature))
        soft_labels_loss = soft_labels_loss * self.temperature**2

        loss = hard_labels_loss + self.alpha * soft_labels_loss
        return loss


class DistillationKLDivLoss:
    def __init__(self,
                 temperature: int=1,
                 alpha: float=0.7,
                 ignore_index: int = -100
                 ) -> None:
        self.standard_targets_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.soft_targets_loss = nn.KLDivLoss(reduction='batchmean')
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.temperature = temperature
        self.alpha = alpha
        self.ignore_index = ignore_index

    def __call__(self, student_logits, teacher_logits, target):
        # Resize teacher logits to match the size of student logits
        resized_teacher_logits = resize_outputs(teacher_logits, student_logits.shape[2:])
        # Resize student logits to match the target size
        resized_student_logits = resize_outputs(student_logits, target.shape[-2:])

        # Standard CrossEntropyLoss
        hard_labels_loss = self.standard_targets_loss(resized_student_logits, target)

        # Distillation loss
        soft_labels_loss = self.soft_targets_loss(self.log_softmax(student_logits / self.temperature),
                                                  self.softmax(resized_teacher_logits / self.temperature))
        soft_labels_loss = soft_labels_loss * self.temperature**2
        soft_labels_loss = soft_labels_loss / (student_logits.shape[2] * student_logits.shape[3])

        loss = hard_labels_loss + self.alpha * soft_labels_loss

        return loss



### Feature-based losses ###


class FeatureMapDistillationMSELoss:
    def __init__(self) -> None:
        self.feature_distillation_loss = nn.MSELoss()

    def __call__(self, student_hidden_layers, teacher_hidden_layers):
        loss = 0

        if len(student_hidden_layers) != len(teacher_hidden_layers):
            raise ValueError("Student and Teacher must be the same number of layers")

        for student_layer, teacher_layer in zip(student_hidden_layers, teacher_hidden_layers):
            if student_layer.shape != teacher_layer.shape:
                # Cut teacher layer to match student layer shape
                student_layer_shape = student_layer.shape
                teacher_layer = teacher_layer[:, :student_layer_shape[1], :student_layer_shape[2], :student_layer_shape[3]]

            loss += self.feature_distillation_loss(student_layer, teacher_layer)
        return loss


class FeatureMapDistillationKLDivLoss:
    def __init__(self) -> None:
        self.feature_distillation_loss = nn.KLDivLoss(reduction='batchmean')
        self.softmax = nn.Softmax(dim=1)

    def __call__(self, student_hidden_layers, teacher_hidden_layers):
        loss = 0

        if len(student_hidden_layers) != len(teacher_hidden_layers):
            raise ValueError("Student and Teacher must be the same number of layers")

        for student_layer, teacher_layer in zip(student_hidden_layers, teacher_hidden_layers):
            student_activations = self.softmax(student_layer)
            teacher_activations = self.softmax(teacher_layer)
            
            loss += self.feature_distillation_loss(student_activations, teacher_activations)
        return loss
