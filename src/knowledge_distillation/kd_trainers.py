import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers.data.data_collator import DataCollator
from typing import Callable, Union, Optional, Dict, Tuple, List
from transformers import (
                          Trainer,
                          PreTrainedModel,
                          TrainerCallback,
                          TrainingArguments,
                          PreTrainedTokenizerBase,
                          EvalPrediction
                          )


class KnowledgeDistillationTrainer(Trainer):
    """Subclass of the Trainer class which implements the distillation training loop.
    """
    def __init__(self,
                 student_model: Union[PreTrainedModel, nn.Module],
                 teacher_model: Union[PreTrainedModel, nn.Module],
                 distillation_loss: Callable,
                 args: TrainingArguments = None,
                 data_collator: Optional[DataCollator] = None,
                 train_dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 model_init: Optional[Callable[[], PreTrainedModel]] = None,
                 compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
                 callbacks: Optional[List[TrainerCallback]] = None,
                 optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
                 preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 ) -> None:
        """Initializes the KnowledgeDistillationTrainer class. Overrides the `__init__` method of the Trainer class (check it for more details and argument descriptions).

        Args:
            student_model (Union[PreTrainedModel, nn.Module]): student model which will be trained using distillation
            teacher_model (Union[PreTrainedModel, nn.Module]): teacher model used for distillation training
            distillation_loss (Callable): function which calculates the distillation loss
        """
        super().__init__(student_model,
                         args,
                         data_collator,
                         train_dataset,
                         eval_dataset,
                         tokenizer,
                         model_init,
                         compute_metrics,
                         callbacks,
                         optimizers,
                         preprocess_logits_for_metrics
                         )

        self.teacher_model = teacher_model.to(self.args.device)
        self.teacher_model.eval()
        self.distillation_loss = distillation_loss

    def compute_loss(self,
                     student_model: nn.Module,
                     inputs: torch.Tensor,
                     return_outputs: bool = False
                     ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Computes the distillation loss for the given student model, teacher model and inputs.
        Overrides the `compute_loss` method of the Trainer class.

        Args:
            student_model (nn.Module): student model
            inputs (torch.Tensor): inputs to the student and teacher model
            return_outputs (bool, optional): denotes if the model outputs should be returned. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: calculated distillation loss or a tuple with the loss and the model outputs
        """
        labels = inputs.pop("labels")

        student_outputs = student_model(**inputs, output_hidden_states=True, output_attentions=True)
        teacher_outputs = self.teacher_model(**inputs, output_hidden_states=True, output_attentions=True)
        
        student_logits = student_outputs.logits
        teacher_logits = teacher_outputs.logits

        # Resize outputs to match the labels
        student_logits = self.resize_outputs(student_logits, labels.shape[-2:])
        teacher_logits = self.resize_outputs(teacher_logits, labels.shape[-2:])

        loss = self.distillation_loss(student_logits, teacher_logits, labels)
        return (loss, student_outputs) if return_outputs else loss

    def resize_outputs(self,
                       output: torch.Tensor,
                       output_size: Tuple[int, int]
                       ) -> torch.Tensor:
        """Resizes the outputs to a given size to match the labels.

        Args:
            output (torch.Tensor): model output tensor to be resized
            output_size (Tuple[int, int]): desired size

        Returns:
            torch.Tensor: resized output tensor
        """
        output_resized = nn.functional.interpolate(
                                output,
                                size=output_size, # (height, width)
                                mode='bilinear',
                                align_corners=False
                                )
        return output_resized
