import argparse
import torch
from pathlib import Path
import sys

sys.path.append('..')

from utils.helpers import available_torch_device


def create_dataset_config() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Image dataset config parser',  add_help=False)

    train_data_path = Path('data/Mapillary_vistas_dataset/training/images')
    train_labels_path = Path('data/Mapillary_vistas_dataset/training/labels')
    val_data_path = Path('data/Mapillary_vistas_dataset/validation/images')
    val_labels_path = Path('data/Mapillary_vistas_dataset/validation/labels')
    json_class_names_file_path = Path('data/Mapillary_vistas_dataset/classes.json')

    # Paths
    parser.add_argument('--train_data_path', type=Path,
                        help='Path to directory containing the training data',
                        default=train_data_path)

    parser.add_argument('--train_labels_path', type=Path,
                        help='Path to directory containing the training labels',
                        default=train_labels_path)
    
    parser.add_argument('--val_data_path', type=Path,
                        help='Path to directory containing the validation data',
                        default=val_data_path)

    parser.add_argument('--val_labels_path', type=Path,
                        help='Path to directory containing the validation labels',
                        default=val_labels_path)

    parser.add_argument('--json_class_names_file_path', type=Path,
                        help='Path to a file containing class names',
                        default=json_class_names_file_path)

    return parser


def create_pipeline_config():
    parser = argparse.ArgumentParser(description='Segmentation model pipeline config parser',  add_help=False)
    
    model_checkpoint = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    
    # Model
    parser.add_argument('--model_checkpoint', type=str,
                        help='Model checkpoint version',
                        default=model_checkpoint)
    
    # General
    parser.add_argument('--img_width', type=int, default=512, help='Desired width of the image')
    parser.add_argument('--img_height', type=int, default=512, help='Desired height of the image')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--device', type=available_torch_device, choices=[torch.device('cpu'), torch.device('cuda'), torch.device('mps')], help='Device used for computation', default='cpu')  # ['cpu', 'cuda', 'mps']

    # Transformations
    data_transformations = parser.add_argument_group()
    data_transformations.add_argument('--seed', type=int, default=50, help='Randomness seed for data transformations')
    data_transformations.add_argument('--num_transforms', type=int, default=2, help='Number of random transformations to apply')
    data_transformations.add_argument('--max_rotation_angle', type=int, default=10, help='Max rotation angle')
    data_transformations.add_argument('--padding', type=int, default=20, help='Padding pixels')
    data_transformations.add_argument('--distortion_factor', type=float, default=0.05, help='Perspective transform factor')
    data_transformations.add_argument('--crop_factor', type=float, default=0.9, help='Image center crop factor')
    data_transformations.add_argument('--sharpness_factor', type=float, default=0.5, help='Default image sharpness factor')
    data_transformations.add_argument('--void_class_id', type=int, default=65, help='Value with which to fill the blank pixels after transformations')
    
    return parser


def parse_train_config() -> argparse.ArgumentParser:
    dataset_config = create_dataset_config()
    pipeline_config = create_pipeline_config()

    parser = argparse.ArgumentParser(description='Training script config parser',
                                     parents=[dataset_config, pipeline_config])
    
    return parser.parse_args()


def parse_evaluation_config() -> argparse.ArgumentParser:
    dataset_config = create_dataset_config()
    pipeline_config = create_pipeline_config()

    parser = argparse.ArgumentParser(description='Evaluation script config parser',
                                     parents=[dataset_config, pipeline_config])

    return parser.parse_args()
