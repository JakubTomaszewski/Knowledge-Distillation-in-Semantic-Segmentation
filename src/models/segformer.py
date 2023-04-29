from os import PathLike
from transformers import PreTrainedModel, SegformerForSemanticSegmentation
from typing import Optional, Mapping, Union


def create_segformer_model_for_train(
                                    model_checkpoint: Union[str, PathLike],
                                    num_classes: int,
                                    id2label: Optional[Mapping] = None,
                                    void_class_id: int = None
                                    ) -> PreTrainedModel:
    """Factory function which initializes a SegformerForSemanticSegmenation model for training.

    Args:
        model_checkpoint (Union[str, PathLike]): checkpoint path or model name.
        num_classes (int): number of classes.
        id2label (Optional[Mapping], optional): class id to class name mapping.
        void_class_id (int, optional): class id not included in the loss calculation.

    Returns:
        PreTrainedModel: SegformerForSemanticSegmentation model.
    """
    label2id = None
    if id2label is not None:
        label2id = {label: class_id for class_id, label in id2label.items()}

    model = SegformerForSemanticSegmentation.from_pretrained(
            model_checkpoint,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            id2label=id2label,
            label2id=label2id,
        )
    
    if void_class_id is not None:
        model.config.semantic_loss_ignore_index = void_class_id

    return model


def create_segformer_model_for_inference(
                                        model_checkpoint: Union[str, PathLike],
                                        id2label: Optional[Mapping] = None
                                        ) -> PreTrainedModel:
    """Factory function which initializes a SegformerForSemanticSegmenation model for inference.

    Args:
        model_checkpoint (Union[str, PathLike]): checkpoint path or model name.
        id2label (Optional[Mapping], optional): class id to class name mapping.

    Returns:
        PreTrainedModel: SegformerForSemanticSegmentation model.
    """
    label2id = None
    if id2label is not None:
        label2id = {label: class_id for class_id, label in id2label.items()}

    model = SegformerForSemanticSegmentation.from_pretrained(
            model_checkpoint,
            id2label=id2label,
            label2id=label2id,
        )
    return model
