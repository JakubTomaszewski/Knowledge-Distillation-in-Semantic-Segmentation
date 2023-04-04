"""Module containing a Mapillary Vistas Image Segmantation Dataset Class and helper functions used in it

Raises:
    OSError: raised when the provided data or labels path does not exist
    OSError: raised when there is no label for a provided data sample filename in the labels dir
    ValueError: raised when data and label shapes do not match
"""

import os
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from torchvision.transforms.functional import pil_to_tensor
from typing import Callable, List, Tuple
import warnings

import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image

from utils.helpers import set_randomness_seed, torch_image_to_numpy, is_image


class MapillaryDataset(Dataset):
    """Class representing a Mapillary Vistas Image Segmantation Dataset

    Args:
        Dataset (torch.utils.data.Dataset): default torch dataset class
    """
    def __init__(self,
                 data_path: Path,
                 labels_path: Path=None,
                 sample_transformation: Callable=None,
                 label_transformation: Callable=None,
                 data_preprocessor: Callable=None,
                 json_class_names_file_path: Path=None) -> None:
        """Initializes the Mapillary Dataset class.

        Args:
            data_path (Path): path to directory with samples
            labels_path (Path): path to directory with labels. Defaults to None - used for testing.
            sample_transformation (Callable): transformation to be applied to each data sample
            label_transformation (Callable): transformation to be applied to each label
            data_preprocessor (Callable): preprocessing operations to be applied to each data sample
            classes_json_path (Path): path to json file containing colors and their corresponding class names

        Initializes:
            sample_filenames (list): list of all sample filenames in provided data_path
            label_filenames (list): list of all label filenames in provided data_path
            id2name (dict): dictionary containing a class id as key and a corresponding class name as value. Initialized if json_class_names_file_path is provided.
            id2color (dict): dictionary containing a class id as key and a corresponding color as value. Initialized if json_class_names_file_path is provided.

        Raises:
            OSError: raised when the provided data or labels path does not exist
        """
        if not data_path.exists():
            raise OSError(f'Path: {data_path} does not exist')

        self._data_path = data_path
        self._labels_path = labels_path
        
        self.sample_filenames = sorted([file for file in os.listdir(self._data_path) if is_image(file)])

        if self._labels_path is not None:
            if not labels_path.exists():
                raise OSError(f'Path: {labels_path} does not exist')
            self._is_test = False
            self.label_filenames = sorted([file for file in os.listdir(self._labels_path) if is_image(file)])
            if len(self.sample_filenames) != len(self.label_filenames):
                warnings.warn('Number of samples does not match the number of labels.')
        else:
            self._is_test = True

        self.data_preprocessor = data_preprocessor
        self.sample_transformation = sample_transformation
        self.label_transformation = label_transformation

        self.id2name = None
        self.id2color = None
        self.num_classes = None

        if json_class_names_file_path is not None:
            self.id2name, self.id2color = self._read_classes_from_json(json_class_names_file_path)
            self.num_classes = len(self.id2name) - 1

    def __len__(self) -> int:
        """Returns the total number of samples.

        Returns:
            int: number of dataset samples
        """
        return len(self.sample_filenames)

    def __getitem__(self, index):  # TODO: Add loading only image for testing (flag .is_test)
        """Loads a data sample and a corresponding label based on the provided index.

        Args:
            index (int): index of the data sample to be loaded

        Raises:
            OSError: raised when there is no label for a provided data sample filename in the labels dir

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: tuple containing transformed sample and label
        """
        if self._is_test:
            return self.get_sample(index)
        else:
            return self.get_sample_and_label(index)

    def get_sample(self, index: int):
        sample_filename = self.sample_filenames[index]
        sample = Image.open(str(self._data_path / sample_filename))
        raw_sample = deepcopy(sample)

        # Checking if transformation should be applied
        if self.sample_transformation is not None:
            sample, _ = self._handle_transformation(sample)

        if self.data_preprocessor is not None:
            inputs = self.data_preprocessor(sample, return_tensors="pt")
            sample = inputs.pixel_values.squeeze()

        return {'pixel_values': sample, 'raw_img': pil_to_tensor(raw_sample)}
    
    def get_sample_and_label(self, index: int):
        sample_filename = self.sample_filenames[index]
        label_filename = self.label_filenames[index]

        if label_filename is None:
            raise OSError(f'No label for such file {sample_filename}')

        sample = Image.open(str(self._data_path / sample_filename))
        label = Image.open(str(self._labels_path / label_filename))
        # self._validate_data(sample, label)

        # Checking if transformation should be applied
        if self.sample_transformation is not None or self.label_transformation is not None:
            sample, label = self._handle_transformation(sample, label)

        if self.data_preprocessor is not None:
            inputs = self.data_preprocessor(sample, label, return_tensors="pt")
            sample = inputs.pixel_values.squeeze()
            label = inputs.labels.squeeze()

        return {'pixel_values': sample, 'labels': label}

    def get_color(self, class_id: int) -> List:
        """Returns the corresponding color for a given class_id.

        Args:
            class_id (int): class id

        Raises:
            ValueError: if a json file with classes has not been provided during initialization

        Returns:
            List: array of RGB values representing a color for the corresponding class id (if exists in dict)
            None: if class id was not found in dict
        """
        if self.id2color is not None:
            return list(self.id2color.get(class_id))
        else:
            return None

    def apply_color_mask(self, img: np.ndarray):
        masked_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        
        for label_id in self.id2color.keys():
            masked_img[img == label_id] = self.get_color(label_id)
        return masked_img

    def get_label_distribution(self):
        print('Counting classes...')
        class_counts = defaultdict(int)

        for label_filename in tqdm(self.label_filenames):
            label = np.array(Image.open(str(self._labels_path / label_filename)))
            class_ids = np.unique(label)
            for class_id in class_ids:
                class_counts[int(class_id)] += 1
        return class_counts

    def _read_classes_from_json(self, json_path: Path) -> Tuple[dict]:
        """Loads a json containing class ids as keys and their corresponding class names and color labels as values.

        Args:
            json_path (Path): path to json file

        Returns:
            Tuple[dict]: 2 dictionaries:
                * id2name - mapping label id's to class names
                * id2color - mapping label id's to colors
        """
        with open(json_path, 'r', encoding='utf-8') as json_file:
            classes_dict = json.load(json_file).items()
            id2name = {int(k): v['name'] for k, v in classes_dict}
            id2color = {int(k): v['color'] for k, v in classes_dict}
            return id2name, id2color

    def _read_classes_from_mapillary_json(self, json_path: Path) -> dict:
        """Loads a json containing color labels and their corresponding class names.

        Args:
            json_path (Path): path to json file containing class ids and their corresponding colors and class names

        Returns:
            dict[Tuple: str]: dictionary containing a class id as key and a dict of corresponding color and class name as value
        """
        with open(json_path, 'r', encoding='utf-8') as json_file:
            labels = json.load(json_file)['labels']
            color_class_dict = {}

            for label_id, label in enumerate(labels):
                color_class_dict[label_id] = {
                    "color": tuple(label['color']),
                    "label": label['name'],
                    "name": label['readable']
                }
            return color_class_dict

    def _validate_data(self,
                       sample: torch.Tensor,
                       label: torch.Tensor):
        """Validates data and label correctness.

        Args:
            sample (torch.Tensor): _description_
            label (torch.Tensor): _description_

        Raises:
            ValueError: raised when data and label shapes do not match
        """
        if sample.size()[1:] != label.size()[1:]:
            raise ValueError(f'Sample and Label dimenstions do not match.\nSample: {sample.size()[1:]}\nLabel: {label.size()[1:]}')
        # TODO: further validation

    def _handle_transformation(self,
                               sample: torch.Tensor,
                               label: torch.Tensor=None
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Handles transformation both for the sample and the label.

        Args:
            sample (torch.Tensor): data sample
            label (torch.Tensor): label

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: tuple containing transformed sample and label
        """
        if self.sample_transformation is not None and self.label_transformation is not None:
            sample, label = self._transform_sample_and_label(sample, label)
        else:
            if self.sample_transformation is not None:
                sample = self.sample_transformation(sample)
            if self.label_transformation:
                label = self.label_transformation(label)
        return sample, label

    def _transform_sample_and_label(self,
                                   sample: torch.Tensor,
                                   label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transforms both sample and label while ensuring the same random transformation parameters.
        Generates a random seed and applies it before performing data and label transformation.

        Args:
            sample (torch.Tensor): data sample
            label (torch.Tensor): label

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: tuple containing transformed sample and label
        """
        seed = np.random.randint(2147483647)
        # Data
        set_randomness_seed(seed)
        sample = self.sample_transformation(sample)

        # Label
        set_randomness_seed(seed)
        label = self.label_transformation(label)
        return sample, label

    @staticmethod
    def load_image(path: str): # TODO: Move to helpers
        """Loads an image from a given path as a torch.Tensor.

        Args:
            path (str): path to image

        Returns:
            torch.Tensor: loaded image
        """
        print(f'loading image from path {path}')  # TODO: logger
        return read_image(path)

    @staticmethod
    def display_torch_image(image: torch.Tensor): # TODO: Move to helpers
        """Displays a torch.Tensor image using matplotlib.

        Args:
            image (torch.Tensor): image in the form of torch.Tensor(channels, height, width) 
        """
        numpy_img = torch_image_to_numpy(image)
        plt.imshow(numpy_img)
        plt.show()

    @staticmethod
    def display_numpy_image(image: np.ndarray): # TODO: Move to helpers
        """Displays a np.ndarray image using matplotlib.

        Args:
            image (np.ndarray): image in the form of np.ndarray(height, width, channels) 
        """
        plt.imshow(image)
        plt.show()
