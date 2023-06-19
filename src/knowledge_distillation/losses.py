import torch.nn as nn
from .utils import resize_outputs


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
