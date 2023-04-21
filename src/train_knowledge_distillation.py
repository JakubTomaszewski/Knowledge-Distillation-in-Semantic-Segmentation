import torch
import torch.nn as nn
from copy import deepcopy
from typing import List, Callable
from argparse import Namespace

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from transformers import TrainingArguments, SchedulerType, TrainerCallback
from transformers.training_args import OptimizerNames
from transformers.integrations import TensorBoardCallback

from knowledge_distillation import KnowledgeDistillationTrainer, DistillationKLDivLoss
from config.configs import parse_kd_train_config
from models.segformer import create_segformer_model_for_train, create_segformer_model_for_inference
from utils.metrics import Evaluator
from data_processing.mapillary_dataset import MapillaryDataset
from data_processing.pipelines.transformation_pipelines import (
                                                      create_data_transformation_pipeline,
                                                      create_label_transformation_pipeline,
                                                      create_evaluation_data_transformation_pipeline,
                                                      create_evaluation_label_transformation_pipeline
                                                      )
from data_processing.pipelines.processing_pipelines import (
                                                  create_data_preprocessing_pipeline,
                                                  create_prediction_postprocessing_pipeline
                                                  )


def configure_tensorboard_logger(config: Namespace):
    return SummaryWriter(
        log_dir=config.tensorboard_log_dir,
        comment=config.student_model_checkpoint+'_KD'
        )


def create_training_args(config: Namespace):
    return TrainingArguments(
        run_name=config.student_model_checkpoint,
        output_dir=config.output_dir,
        overwrite_output_dir=config.overwrite_output_dir,
        seed=config.seed,
        no_cuda=config.device != torch.device('cuda'),
        # use_mps_device=config.device == torch.device('mps'),

        # ------ Logging & Saving ------ #
        logging_strategy='epoch',
        logging_dir=config.tensorboard_log_dir,
        report_to='tensorboard',
        save_strategy='epoch',
        save_total_limit=config.num_checkpoints_to_save,

        # ------ Train hyperparameters: ------ #
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        # Optimizer
        optim=OptimizerNames.ADAMW_TORCH,
        learning_rate=config.learning_rate,
        adam_beta1=config.optimizer_betas[0],
        adam_beta2=config.optimizer_betas[1],
        warmup_ratio=1e-6,
        warmup_steps=1500,
        # Scheduler
        lr_scheduler_type=SchedulerType.POLYNOMIAL,

        # ------ Eval params: ------ #
        evaluation_strategy='epoch',
        per_device_eval_batch_size=config.val_batch_size,
        eval_accumulation_steps=500
    )


def create_kd_trainer(student_model: nn.Module,
                      teacher_model: nn.Module,
                      loss_func: Callable,
                      training_args: TrainingArguments,
                      train_dataset: Dataset,
                      eval_dataset: Dataset,
                      prediction_postprocessing_pipeline: Callable,
                      metric: Callable,
                      callbacks: List[TrainerCallback] = []
                      ) -> KnowledgeDistillationTrainer:
    return KnowledgeDistillationTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        distillation_loss=loss_func,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        preprocess_logits_for_metrics=prediction_postprocessing_pipeline,
        compute_metrics=metric,
        callbacks=callbacks,
    )


if __name__ == '__main__':
    # Config
    train_config = parse_kd_train_config()
    evaluation_config = parse_kd_train_config()
    
    img_shape = (train_config.img_height, train_config.img_width)

    # Data transformations
    data_transformation_pipeline = create_data_transformation_pipeline(train_config)
    label_transformation_pipeline = create_label_transformation_pipeline(train_config)
    evaluation_data_transformation_pipeline = create_evaluation_data_transformation_pipeline(evaluation_config)
    evaluation_label_transformation_pipeline = create_evaluation_label_transformation_pipeline(evaluation_config)
    
    # Data processing
    data_preprocessing_pipeline = create_data_preprocessing_pipeline(train_config)
    evaluation_data_preprocessing_pipeline = create_data_preprocessing_pipeline(evaluation_config)
    prediction_postprocessing_pipeline = create_prediction_postprocessing_pipeline(img_shape)

    # Datasets
    train_dataset = MapillaryDataset(train_config.train_data_path,
                                     train_config.train_labels_path,
                                     sample_transformation=data_transformation_pipeline,
                                     label_transformation=label_transformation_pipeline,
                                     data_preprocessor=data_preprocessing_pipeline,
                                     json_class_names_file_path=train_config.json_class_names_file_path)

    eval_dataset = MapillaryDataset(evaluation_config.val_data_path,
                                    evaluation_config.val_labels_path,
                                    sample_transformation=evaluation_data_transformation_pipeline,
                                    label_transformation=evaluation_label_transformation_pipeline,
                                    data_preprocessor=evaluation_data_preprocessing_pipeline,
                                    json_class_names_file_path=evaluation_config.json_class_names_file_path)

    # Metrics
    evaluator = Evaluator(class_labels=train_dataset.id2name.keys(),
                          ignore_index=train_config.void_class_id)

    # Class mappings
    id2name = deepcopy(train_dataset.id2name)
    id2name.pop(train_config.void_class_id)

    # Model
    student_model = create_segformer_model_for_train(train_config.student_model_checkpoint,
                                                     train_dataset.num_classes,
                                                     id2label=id2name,
                                                     void_class_id=train_config.void_class_id
                                                     )
    
    teacher_model = create_segformer_model_for_inference(train_config.teacher_model_checkpoint,
                                                         id2name
                                                         )

    # Logging
    tb_writer = configure_tensorboard_logger(train_config)

    # Callbacks
    train_callbacks = [
        TensorBoardCallback(tb_writer)
        ]

    # Loss
    loss_func = DistillationKLDivLoss(train_config.temperature,
                                      train_config.alpha,
                                      ignore_index=train_config.void_class_id)

    # Trainer
    training_args = create_training_args(train_config)
    trainer = create_kd_trainer(student_model,
                                teacher_model,
                                loss_func,
                                training_args,
                                train_dataset,
                                eval_dataset,
                                prediction_postprocessing_pipeline,
                                metric=evaluator.compute_metrics,
                                callbacks=train_callbacks
                                )
    trainer.train()
