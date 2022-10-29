"""Module containing a Mapillary Vistas Image Segmantation Dataset Class and helper functions used in it

Raises:
    OSError: raised when the provided data or labels path does not exist
    OSError: raised when there is no label for a provided data sample filename in the labels dir
    ValueError: raised when data and label shapes do not match
"""

import os
import json
from pathlib import Path
from typing import Callable, List, Tuple
import warnings

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.io import read_image

from utils.helpers import set_randomness_seed, torch_image_to_numpy


class MapillaryDataset(Dataset):
    """Class representing a Mapillary Vistas Image Segmantation Dataset

    Args:
        Dataset (torch.utils.data.Dataset): default torch dataset class
    """
    def __init__(self,
                 data_path: Path,
                 labels_path: Path,
                 sample_transformation: Callable=None,
                 label_transformation: Callable=None,
                 json_class_names_file_path: Path=None) -> None:
        """Initializes the Mapillary Dataset class.

        Args:
            data_path (Path): path to directory with samples
            labels_path (Path): path to directory with labels
            sample_transformation (Callable): transformation to be applied to each data sample
            label_transformation (Callable): transformation to be applied to each label
            classes_json_path (Path): path to json file containing colors and their corresponding class names

        Raises:
            OSError: raised when the provided data or labels path does not exist
        """
        if not data_path.exists():
            raise OSError(f'Path: {data_path} does not exist')
        if not labels_path.exists():
            raise OSError(f'Path: {labels_path} does not exist')

        self._data_path = data_path
        self._labels_path = labels_path
        self.sample_filenames = os.listdir(self._data_path)
        self.label_filenames = os.listdir(self._labels_path)

        if len(self.sample_filenames) != len(self.label_filenames):
            warnings.warn('Number of samples does not match the number of labels.')

        self.sample_transformation = sample_transformation
        self.label_transformation = label_transformation

        if json_class_names_file_path is not None:
            self.color_classname_dict = self._load_classes_from_json(json_class_names_file_path)
        else:
            self.color_classname_dict = None

    def __len__(self) -> int:
        """Returns the total number of samples.

        Returns:
            int: number of dataset samples
        """
        return len(self.sample_filenames)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loads a data sample and a corresponding label based on the provided index.

        Args:
            index (int): index of the data sample to be loaded

        Raises:
            OSError: raised when there is no label for a provided data sample filename in the labels dir

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: tuple containing transformed sample and label
        """
        sample_filename = self.sample_filenames[index]
        label_filename = get_corresponding_filename(sample_filename, self.label_filenames)

        if label_filename is None:
            raise OSError(f'No label for such file {sample_filename}')

        sample = self.load_image(str(self._data_path / sample_filename))
        label = self.load_image(str(self._labels_path / label_filename))
        self._validate_data(sample, label)

        # Checking if transformation should be applied
        if self.sample_transformation is not None or self.label_transformation is not None:
            sample, label = self._handle_transformation(sample, label)
        return sample, label

    def color_to_class(self, color: Tuple[int]):
        """Returns the corresponding class name for a color provided.

        Args:
            color (Tuple[int]): 3 element int tuple representing an RGB color

        Raises:
            ValueError: if a json file with class names has not been provided during initialization

        Returns:
            str: corresponding class name (if exists in dict)
            None: if color class was not found dict
        """
        if self.color_classname_dict is not None:
            return self.color_classname_dict.get(color)
        else:
            raise ValueError('File with the class names has not been specified')

    def _load_classes_from_json(self, json_path: Path) -> dict:
        """Loads a json containing color labels and their corresponding class names.

        Args:
            json_path (Path): path to json file containing colors and their corresponding class names

        Returns:
            dict[Tuple: str]: dictionary containing a color tuple as a key, and the corresponding class name as value
        """
        with open(json_path, 'r', encoding='utf-8') as json_file:
            labels = json.load(json_file)['labels']
            color_class_dict = {}

            for label in labels:
                color_class_dict.update(self._extract_color_classname_pair(label))
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
                               label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def _extract_color_classname_pair(self, label_dict):
        return {tuple(label_dict['color']): label_dict['readable']}

    @staticmethod
    def load_image(path: str):
        """Loads an image from a given path as a torch.Tensor.

        Args:
            path (str): path to image

        Returns:
            torch.Tensor: loaded image
        """
        print(f'loading image from path {path}')  # TODO: logger
        return read_image(path)

    @staticmethod
    def display_image(image: torch.Tensor):
        """Displays a torch.Tensor image using matplotlib.

        Args:
            image (torch.Tensor): image in the form of torch.Tensor(channels, height, width) 
        """
        numpy_img = torch_image_to_numpy(image)
        rgb_img = cv2.cvtColor(numpy_img, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_img)
        plt.show()

    def get_sample(self, index):
        # TODO: loads and returns an image sample based on the provided index.
        pass

    def get_label(self, index):
        # TODO: loads and returns an label based on the provided index.
        pass


# Helper functions

def get_first_filename_match(filenames, pattern_filename):
    """_summary_

    Args:
        filenames (list): _description_
        pattern_filename (str): _description_

    Returns:
        str: the matched filename
        None: if no filename was matched
    """
    pattern_name = os.path.splitext(pattern_filename)[0]
    matched_filename = next(filter(lambda x: os.path.splitext(x)[0] == pattern_name, filenames), None)
    return matched_filename


def get_corresponding_filename(pattern_filename: str, filenames: List[str]) -> str:
    """Finds a complete filename from the list corresponding to the provided pattern filename.
    Checks if the filenames match (omits the extensions). Then returns the complete filename (with the file extension).

    Args:
        pattern_filename (str): complete pattern filename (with extension)

    Returns:
        str: the matched complete filename
        None: if no filename was matched
    """
    pattern_name = os.path.splitext(pattern_filename)[0]
    filter_condition = filter(lambda x: os.path.splitext(x)[0] == pattern_name, filenames)
    matched_filename = next(filter_condition, None)
    return matched_filename
