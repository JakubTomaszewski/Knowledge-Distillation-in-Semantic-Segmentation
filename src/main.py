from pip import main
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import random
from torchvision.transforms import RandomRotation

# random.seed(10)
# torch.manual_seed(10)

from data_processing.mapillary_dataset import MapillaryDataset
from data_processing.data_transformations import create_data_transformation_pipeline, create_label_transformation_pipeline


IMG_SIZE = (400, 600)
_MAX_ROTATION_ANGLE = 15
_PADDING=25
_DISTORTION_FACTOR=0.2


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device


if __name__ == '__main__':
    data_path = Path('../data/Mapillary_vistas_dataset/training/images')  # load from ArgParse
    labels_path = Path('../data/Mapillary_vistas_dataset/training/labels')

    device = get_device()
    print(f'Available device: {device}')

    data_transformation_pipeline = create_data_transformation_pipeline(IMG_SIZE, _MAX_ROTATION_ANGLE, _PADDING, _DISTORTION_FACTOR)
    label_transformation_pipeline = create_label_transformation_pipeline(IMG_SIZE, _MAX_ROTATION_ANGLE, _PADDING, _DISTORTION_FACTOR)

    m_dataset = MapillaryDataset(data_path,
                                 labels_path,
                                 sample_transformation=data_transformation_pipeline,
                                 label_transformation=label_transformation_pipeline)
    
    m_dataloader = DataLoader(m_dataset, batch_size=1, shuffle=True)

    img, label = m_dataset[0]
    
    print(type(img))
    print(img.shape)
    m_dataset.display_image(img)
    m_dataset.display_image(label)

    for batch_num, batch in enumerate(m_dataloader):
        img, label = batch
        m_dataset.display_image(img[0])
        m_dataset.display_image(label[0])
        if batch_num >= 0:
            break
