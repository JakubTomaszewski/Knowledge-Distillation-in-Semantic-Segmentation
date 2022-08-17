import sys
from pathlib import Path
from torch.utils.data import DataLoader

from data_processing.mapillary_dataset import MapillaryDataset
from data_processing.data_transformations import (
                                                 create_data_transformation_pipeline,
                                                 create_label_transformation_pipeline
                                                 )

sys.path.append('..')

from utils.helpers import set_randomness_seed, get_device
from config.dataset_config import parse_dataset_config


if __name__ == '__main__':
    dataset_config = parse_dataset_config()

    data_path = Path('../data/Mapillary_vistas_dataset/training/images')  # load from ArgParse
    labels_path = Path('../data/Mapillary_vistas_dataset/training/labels')

    set_randomness_seed(dataset_config.seed)
    device = get_device()
    print(f'Available device: {device}')

    data_transformation_pipeline = create_data_transformation_pipeline(dataset_config)
    label_transformation_pipeline = create_label_transformation_pipeline(dataset_config)

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
