import argparse
from pathlib import Path


def parse_dataset_config() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Image dataset config parser')

    train_data_path = Path('../data/Mapillary_vistas_dataset/training/images')
    train_labels_path = Path('../data/Mapillary_vistas_dataset/training/labels')
    json_class_names_file_path = Path('../data/Mapillary_vistas_dataset/classes.json')

    # Paths
    parser.add_argument('--train_data_path', type=Path,
                        help='Path to directory containing the training data',
                        default=train_data_path)

    parser.add_argument('--train_labels_path', type=Path,
                        help='Path to directory containing the training labels',
                        default=train_labels_path)

    parser.add_argument('--json_class_names_file_path', type=Path,
                        help='Path to a file containing class names',
                        default=json_class_names_file_path)

    # General
    parser.add_argument('--img_width', type=int, default=600, help='Desired width of the image')
    parser.add_argument('--img_height', type=int, default=400, help='Desired height of the image')

    # Transformations
    data_transformations = parser.add_argument_group()
    data_transformations.add_argument('--seed', type=int, default=50, help='Randomness seed for data transformations')
    data_transformations.add_argument('--num_transformers', type=int, default=3, help='Number of random transformations to apply')
    data_transformations.add_argument('--max_rotation_angle', type=int, default=10, help='Max rotation angle')
    data_transformations.add_argument('--padding', type=int, default=20, help='Padding pixels')
    data_transformations.add_argument('--distortion_factor', type=float, default=0.1, help='Perspective transform factor')
    data_transformations.add_argument('--crop_factor', type=float, default=0.9, help='Image center crop factor')
    data_transformations.add_argument('--sharpness_factor', type=float, default=0.5, help='Default image sharpness factor')
    data_transformations.add_argument('--fill_value', type=int, default=0, help='Value with which to fill the blank pixels after transformations')
    
    return parser.parse_args()
