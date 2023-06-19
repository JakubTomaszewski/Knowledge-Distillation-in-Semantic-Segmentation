"""
Script for visualizing dataset samples.
"""

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils.helpers import set_randomness_seed, torch_image_to_numpy
from config.configs import parse_train_config
from data_processing.mapillary_dataset import MapillaryDataset
from data_processing.pipelines.transformation_pipelines import (
                                                 create_data_transformation_pipeline,
                                                 create_label_transformation_pipeline
                                                 )
from data_processing.pipelines.processing_pipelines import (
                                                create_data_preprocessing_pipeline,
                                                create_prediction_postprocessing_pipeline
                                                )


if __name__ == '__main__':
    # Config
    train_config = parse_train_config()
    img_shape = (train_config.img_height, train_config.img_width)
    train_config.crop_size = img_shape

    set_randomness_seed(train_config.seed)

    # Data transformations
    data_transformation_pipeline = create_data_transformation_pipeline(train_config)
    label_transformation_pipeline = create_label_transformation_pipeline(train_config)

    # Data processing
    data_preprocessing_pipeline = create_data_preprocessing_pipeline(train_config)
    prediction_postprocessing_pipeline = create_prediction_postprocessing_pipeline(img_shape)

    # Dataset
    m_dataset = MapillaryDataset(train_config.train_data_path,
                               train_config.train_labels_path,
                               data_preprocessor=data_preprocessing_pipeline,
                               json_class_names_file_path=train_config.json_class_names_file_path)

    # Dataloader
    m_dataloader = DataLoader(m_dataset, batch_size=1, shuffle=True)

    for batch_num, batch in enumerate(m_dataloader):
        img, label = batch['pixel_values'], batch['labels']
        img = torch_image_to_numpy(img.squeeze())

        # Display img and label
        fig, ax = plt.subplots(2, figsize=(8, 8))
        ax[0].imshow(img.squeeze())
        ax[0].set_title('Sample')
        ax[1].imshow(label.squeeze())
        ax[1].set_title('Label')
        plt.show()

        if batch_num >= 1:
            break
