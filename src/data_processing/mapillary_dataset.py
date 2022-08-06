import os
import random
from pathlib import Path
from typing import List, Tuple
import warnings

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.io import read_image


class MapillaryDataset(Dataset):
    def __init__(self, data_path: Path, labels_path: Path, transformation=None) -> None:
        """Initializes the Mapillary Dataset class.

        Args:
            data_path (Path): path to directory with samples
            labels_path (Path): path to directory with labels

        Raises:
            ValueError: _description_
        """
        if not data_path.exists():
            raise OSError(f'Path: {data_path} does not exist')
        if not labels_path.exists():
            raise OSError(f'Path: {labels_path} does not exist')

        self.data_path = data_path
        self.labels_path = labels_path
        self.sample_filenames = os.listdir(self.data_path)
        self.label_filenames = os.listdir(self.labels_path)

        if len(self.sample_filenames) != len(self.label_filenames):
            warnings.warn('Number of samples does not match the number of labels.')
        self.transformation = transformation

    def get_corresponding_label_filename(self, sample_filename):
        """_summary_

        Args:
            filenames (list): _description_
            pattern_filename (str): _description_

        Returns:
            str: the matched filename
            None: if no filename was matched
        """
        pattern_name = os.path.splitext(sample_filename)[0]
        matched_filename = next(filter(lambda x: os.path.splitext(x)[0] == pattern_name, self.label_filenames), None)
        return matched_filename

    def load_image(self, path: str):
        print(f'loading image from path {path}')  # logger?
        return read_image(path)

    def __len__(self):
        """Returns the total number of samples.

        Returns:
            int: number of dataset samples
        """
        return len(self.sample_filenames)

    def __getitem__(self, index) -> Tuple:
        # transform it (resize, transform to tensor etc.) - just apply transformation
        # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#transforms
        sample_filename = self.sample_filenames[index]
        label_filename = self.get_corresponding_label_filename(sample_filename)
        if label_filename is None:
            raise ValueError(f'No label for such file {sample_filename}')

        sample = self.load_image(str(self.data_path / sample_filename))
        label = self.load_image(str(self.labels_path / label_filename))
        
        if sample.size()[1:] != label.size()[1:]:  # checking if the image size matches
            raise ValueError(f'Sample and Label dimenstions do not match.\nSample: {sample.size()[1:]}\nLabel: {label.size()[1:]}')

        if self.transformation is not None:
            seed = np.random.randint(2147483647)
            sample, label = self.transform_sample_and_label(sample, label, seed)
            
        return sample, label

    def transform_sample_and_label(self, sample, label, seed):
        """Ensures the same transformations for sample and label

        Args:
            sample (torch.Tensor): _description_
            label (_type_): _description_
            seed (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Seting seed for sample transform
        MapillaryDataset.set_random_seed(seed)
        # Transforming sample
        sample = self.transformation(sample)
        
        # Setting the same seed for label transform
        MapillaryDataset.set_random_seed(seed)
        # Transforming label
        label = self.transformation(label)
        return sample, label

    @staticmethod
    def set_random_seed(seed):
        """Sets a seed for computations performed by torch and random library

        Args:
            seed (int): random seed to be set
        """
        random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def display_image(image):
        """Displays a torch the image in the type of Tensor

        Args:
            image (torch.tensor): image in the form of torch.Tensor(channels, height, width) 
        """
        plt.imshow(image.permute(1, 2, 0))
        plt.show()

    @staticmethod
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

    @staticmethod
    def _create_sample_label_dict(sample_filenames, label_filenames):
        N = len(sample_filenames)
        sample_label_dict = {}
        for i, sample_filename in enumerate(sample_filenames):
            label_filename = MapillaryDataset.get_first_filename_match(label_filenames, sample_filename)
            if label_filename is None:
                print("Missing label for")  # TODO: logger
            sample_label_dict[sample_filename] = label_filename
        return sample_label_dict
