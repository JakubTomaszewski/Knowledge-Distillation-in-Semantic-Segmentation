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
                # Permute and Interpolate the student layer to match the teacher layer size
                permuted_student_layer = student_layer.permute(0, 2, 3, 1)
                current_shape = torch.IntTensor(list(teacher_layer.shape))
                desired_shape = torch.Size(torch.index_select(current_shape, dim=0, index=torch.IntTensor([3, 1])))

                interpolated_student_layer = F.interpolate(permuted_student_layer, size=desired_shape, mode='bilinear', align_corners=False)
                student_layer = interpolated_student_layer.permute(0, 3, 1, 2)

            loss += self.feature_distillation_loss(student_layer, teacher_layer)
        return loss
