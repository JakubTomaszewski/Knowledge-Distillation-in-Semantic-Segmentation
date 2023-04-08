import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append('../../src')

from data_processing.pipelines.transformation_pipelines import (
                                                 create_evaluation_data_transformation_pipeline,
                                                 create_evaluation_label_transformation_pipeline
                                                 )
from utils.helpers import set_randomness_seed, torch_image_to_numpy
from config.configs import parse_train_config
from data_processing.mapillary_dataset import MapillaryDataset


CLASS_ID_TO_INSPECT = 10
DATASET_DIR = '../../data/Mapillary_vistas_dataset'


if __name__ == '__main__':
    # Config
    train_config = parse_train_config()
    img_shape = (train_config.img_height, train_config.img_width)
    train_config.crop_size = img_shape

    set_randomness_seed(train_config.seed)

    # Data transformations
    data_transformation_pipeline = create_evaluation_data_transformation_pipeline(train_config)
    label_transformation_pipeline = create_evaluation_label_transformation_pipeline(train_config)

    # Dataset   
    m_dataset = MapillaryDataset(Path(f'{DATASET_DIR}/training/images'),
                                 Path(f'{DATASET_DIR}/training/labels'),
                                 sample_transformation=data_transformation_pipeline,
                                 label_transformation=label_transformation_pipeline,
                                 json_class_names_file_path=Path('../../data/Mapillary_vistas_dataset/classes.json'))

    # Dataloader
    m_dataloader = DataLoader(m_dataset, batch_size=1, shuffle=True)

    class_appearances = 0

    for batch_num, batch in tqdm(enumerate(m_dataloader)):
        img, label = batch['pixel_values'], batch['labels']

        if CLASS_ID_TO_INSPECT not in label:
            continue
        
        class_appearances += 1

        img = torch_image_to_numpy(img.squeeze())
        label = torch_image_to_numpy(label[0]).squeeze()

        class_mask = label == CLASS_ID_TO_INSPECT
        masked_img = img.copy()
        
        color_mask = np.zeros_like(masked_img)
        color_mask[class_mask] = [10, 240, 0]
        
        masked_img = cv2.addWeighted(img, 1, color_mask, 0.5, 0)
        
        # Display img and label
        fig, ax = plt.subplots(1, 2, figsize=(17, 8))
        ax[0].imshow(img)
        ax[0].set_title('Sample')
        ax[1].imshow(masked_img)
        ax[1].set_title('Label')
        plt.tight_layout()
        plt.show()

        if class_appearances >= 5:
            break
