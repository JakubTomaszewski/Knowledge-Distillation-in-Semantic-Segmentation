import os
import torch
import torch.nn as nn
from copy import deepcopy
from typing import List, Callable
from argparse import Namespace

import mlflow
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer, SchedulerType, TrainerCallback
from transformers.training_args import OptimizerNames
from transformers.integrations import MLflowCallback

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


def configure_mlflow_logger(config: Namespace):
    os.environ["MLFLOW_EXPERIMENT_NAME"]=config.model_checkpoint
    os.environ["MLFLOW_FLATTEN_PARAMS"]="1"

    mlflow.create_experiment(config.model_checkpoint, str(config.mlflow_log_dir))
    mlflow.set_tracking_uri(config.mlflow_log_dir)



def create_training_args(config: Namespace):
    return TrainingArguments(
        run_name=config.model_checkpoint,
        output_dir=config.output_dir,
        overwrite_output_dir=config.overwrite_output_dir,
        seed=config.seed,
        no_cuda=config.device != torch.device('cuda'),
        # use_mps_device=config.device == torch.device('mps'),

        # ------ Logging & Saving ------ #
        logging_strategy='epoch',
        logging_dir=config.output_log_dir,
        report_to='mlflow',
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
        eval_accumulation_steps=1000
    )


def create_trainer(model: nn.Module,
                   training_args: TrainingArguments,
                   train_dataset: Dataset,
                   eval_dataset: Dataset,
                   prediction_postprocessing_pipeline: Callable,
                   metric: Callable,
                   callbacks: List[TrainerCallback] = []
                   ) -> Trainer:
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
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
    model = create_segformer_model_for_train(train_config, train_dataset.num_classes, id2name)

    # Logging
    configure_mlflow_logger(train_config)
    train_callbacks = [
        MLflowCallback()
        ]

    # Trainer
    training_args = create_training_args(train_config)
    trainer = create_trainer(model,
                             training_args,
                             train_dataset,
                             eval_dataset,
                             prediction_postprocessing_pipeline,
                             metric=evaluator.compute_metrics,
                             callbacks=train_callbacks
                             )
    trainer.train()
