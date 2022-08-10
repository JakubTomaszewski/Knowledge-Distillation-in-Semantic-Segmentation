from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np

from utils.helpers import set_randomness_seed
from data_processing.mapillary_dataset import MapillaryDataset
from data_processing.data_transformations import (
                                                 create_data_transformation_pipeline,
                                                 create_label_transformation_pipeline
                                                 )


IMG_SIZE = (400, 600)
_MAX_ROTATION_ANGLE = 10
_PADDING=20
_DISTORTION_FACTOR=0.1
_CROP_FACTOR=0.9
_SHARPNESS_FACTOR=0.5


def get_device():
    """Returns the available device for computation.

    Returns:
        torch.device: available device for computation
    """
    compute_device = None
    if torch.cuda.is_available():
        compute_device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        compute_device = torch.device('mps')
    else:
        compute_device = torch.device('cpu')
    return compute_device


if __name__ == '__main__':
    seed = np.random.randint(2147483647)

    data_path = Path('../data/Mapillary_vistas_dataset/training/images')  # load from ArgParse
    labels_path = Path('../data/Mapillary_vistas_dataset/training/labels')

    device = get_device()
    print(f'Available device: {device}')
    print(np.array(IMG_SIZE) * 0.9)

    set_randomness_seed(seed)
    data_transformation_pipeline = create_data_transformation_pipeline(IMG_SIZE, _MAX_ROTATION_ANGLE, _PADDING, _DISTORTION_FACTOR, _CROP_FACTOR, _SHARPNESS_FACTOR, fill_value=0)

    set_randomness_seed(seed)
    label_transformation_pipeline = create_label_transformation_pipeline(IMG_SIZE, _MAX_ROTATION_ANGLE, _PADDING, _DISTORTION_FACTOR, _CROP_FACTOR, _SHARPNESS_FACTOR, fill_value=0)

    print('data_transformation_pipeline:', data_transformation_pipeline)
    print('label_transformation_pipeline:', label_transformation_pipeline)

    m_dataset = MapillaryDataset(data_path,
                                 labels_path,
                                 sample_transformation=data_transformation_pipeline,
                                 label_transformation=label_transformation_pipeline)

    m_dataloader = DataLoader(m_dataset, batch_size=1, shuffle=True)

    img, label = m_dataset[0]

    m_dataset.display_image(img)
    m_dataset.display_image(label)

    for batch_num, batch in enumerate(m_dataloader):
        img, label = batch
        m_dataset.display_image(img[0])
        m_dataset.display_image(label[0])
        if batch_num >= 1:
            break
