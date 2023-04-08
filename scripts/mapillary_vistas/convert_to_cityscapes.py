# Script which converts Mapillary Vistas dataset labels to Cityscapes format (65 -> 19 classes)

import numpy as np
import sys
import json
import os
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import Mapping

sys.path.append('../../src')

from utils.helpers import is_image


CLASS_MAPPINGS_FILE = Path('./mapillary_cityscapes_class_mapping.json')

# Data paths
DATA_DIR = Path('../../data/Mapillary_vistas_dataset')
LABELS_DIR = DATA_DIR / 'validation/labels'

# Output paths
MAPPED_LABELS_DIR = DATA_DIR / 'validation/labels_mapped'


def read_class_mappings(json_file_path) -> dict:
    with open(json_file_path, 'r') as file:
        class_mappings = json.load(file)
    return class_mappings


def map_classes(class_mappings: Mapping) -> None:
    label_filenames = [filename for filename in os.listdir(LABELS_DIR) if is_image(filename)]

    for filename in tqdm(label_filenames):
        label = np.array(Image.open(LABELS_DIR / filename))
        new_label = np.zeros_like(label)

        for class_id in class_mappings.keys():
            new_label[label == int(class_id)] = class_mappings[class_id]

        Image.fromarray(new_label).save(MAPPED_LABELS_DIR / filename)


if __name__ == '__main__':
    mapillary2cityscapes_dict = read_class_mappings(CLASS_MAPPINGS_FILE)

    if not MAPPED_LABELS_DIR.exists():
        os.makedirs(MAPPED_LABELS_DIR)

    map_classes(mapillary2cityscapes_dict)
