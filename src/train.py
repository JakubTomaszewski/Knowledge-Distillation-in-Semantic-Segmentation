"""
Script for training the model.
"""

import torch
import torch.nn as nn
from copy import deepcopy
from typing import List, Callable, Union, Dict
from argparse import Namespace

from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer, SchedulerType, TrainerCallback
from transformers.training_args import OptimizerNames
from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter

from config.configs import parse_train_config
from models.segformer import create_segformer_model_for_train
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
    return SummaryWriter(log_dir=config.tensorboard_log_dir)


def create_training_args(config: Namespace):
    return TrainingArguments(
        run_name=config.model_checkpoint,
        output_dir=config.output_dir,
        overwrite_output_dir=config.overwrite_output_dir,
        seed=config.seed,
        no_cuda=config.device != torch.device('cuda'),

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


def create_trainer(model: nn.Module,
                   training_args: TrainingArguments,
                   train_dataset: Dataset,
                   eval_datasets: Union[Dataset, Dict[str, Dataset]],
                   prediction_postprocessing_pipeline: Callable,
                   metric: Callable,
                   callbacks: List[TrainerCallback] = []
                   ) -> Trainer:
    """Factory function which creates a Hugging Face Trainer.

    Args:
        model (nn.Module): model to train
        training_args (TrainingArguments): model training arguments and hyperparameters
        train_dataset (Dataset): dataset used for training
        eval_datasets (Union[Dataset, Dict[str, Dataset]]): dataset/s used for evaluation
        prediction_postprocessing_pipeline (Callable): pipeline used to process model predictions before evaluation
        metric (Callable): function used to compute metrics
        callbacks (List[TrainerCallback], optional): list of training callbacks. Defaults to [].

    Returns:
        Trainer: initialized Hugging Face Trainer
    """
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        preprocess_logits_for_metrics=prediction_postprocessing_pipeline,
        compute_metrics=metric,
        callbacks=callbacks,
    )


if __name__ == '__main__':
    # Config
    evaluation_config = parse_train_config()
    train_config = parse_train_config()

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
    model = create_segformer_model_for_train(train_config.model_checkpoint,
                                             train_dataset.num_classes,
                                             id2name,
                                             void_class_id=train_config.void_class_id
                                             )

    # Logging
    tb_writer = configure_tensorboard_logger(train_config)

    # Callbacks
    train_callbacks = [
        TensorBoardCallback(tb_writer)
        ]

    # Trainer
    training_args = create_training_args(train_config)
    trainer = create_trainer(model,
                             training_args,
                             train_dataset,
                             eval_datasets=eval_dataset,  # {'val': eval_dataset, 'train': train_dataset}
                             prediction_postprocessing_pipeline=prediction_postprocessing_pipeline,
                             metric=evaluator.compute_metrics,
                             callbacks=train_callbacks
                             )
    trainer.train()
