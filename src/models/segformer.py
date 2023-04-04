from transformers import PreTrainedModel, SegformerForSemanticSegmentation, SegformerConfig
from typing import Optional, Mapping


def create_segformer_model_for_train(config: Mapping,
                           num_classes: int,
                           id2label: Optional[Mapping] = None
                           ) -> PreTrainedModel:
    label2id = None
    if id2label is not None:
        label2id = {label: class_id for class_id, label in id2label.items()}

    model = SegformerForSemanticSegmentation.from_pretrained(
            config.model_checkpoint,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            id2label=id2label,
            label2id=label2id,
        )
    model.config.semantic_loss_ignore_index = config.void_class_id
    return model


def create_segformer_model_for_inference(config: Mapping,
                                         id2label: Optional[Mapping] = None
                                         ) -> PreTrainedModel:
    label2id = None
    if id2label is not None:
        label2id = {label: class_id for class_id, label in id2label.items()}

    model = SegformerForSemanticSegmentation.from_pretrained(
            config.model_checkpoint,
            id2label=id2label,
            label2id=label2id,
        )
    return model
