import torch.nn as nn


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
        # Standard CrossEntropyLoss
        hard_labels_loss = self.standard_targets_loss(student_logits, target)

        # Distillation loss
        soft_labels_loss = self.soft_targets_loss(self.log_softmax(student_logits / self.temperature),
                                                  self.softmax(teacher_logits / self.temperature))
        soft_labels_loss = soft_labels_loss * self.temperature**2

        loss = (1 - self.alpha) * hard_labels_loss + self.alpha * soft_labels_loss
        return loss


class DistillationKLDivLoss:
    def __init__(self,
                 temperature: int=1,
                 alpha: float=0.7,
                 ignore_index: int = -100
                 ) -> None:
        self.standard_targets_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.soft_targets_loss = nn.KLDivLoss()
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.temperature = temperature
        self.alpha = alpha
        self.ignore_index = ignore_index

    def __call__(self, student_logits, teacher_logits, target):
        # Standard CrossEntropyLoss
        hard_labels_loss = self.standard_targets_loss(student_logits, target)

        # Distillation loss
        soft_labels_loss = self.soft_targets_loss(self.log_softmax(student_logits / self.temperature),
                                                  self.softmax(teacher_logits / self.temperature))
        soft_labels_loss = soft_labels_loss * self.temperature**2

        loss = (1 - self.alpha) * hard_labels_loss + self.alpha * soft_labels_loss
        return loss
