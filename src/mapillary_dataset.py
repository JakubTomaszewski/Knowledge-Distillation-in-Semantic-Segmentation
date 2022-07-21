import os
import cv2
from pathlib import Path
from typing import List, Tuple
import warnings

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
        if data_path.exists():
            self.data_path = data_path
        if labels_path.exists():
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
        return len(self.sample_names)

    def __getitem__(self, index) -> Tuple:
        # transform it (resize, transform to tensor etc.) - just apply transformation
        # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#transforms
        sample_filename = self.sample_filenames[index]
        label_filename = self.get_corresponding_label_filename(sample_filename)
        if label_filename is None:
            raise ValueError(f'No label for such file {sample_filename}')

        sample = self.load_image(str(self.data_path / sample_filename))
        label = self.load_image(str(self.labels_path / label_filename))
        
        if self.transformation is not None:
            sample = self.transformation(sample)
        return sample, label

    @staticmethod
    def display_image(image):
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
    def create_sample_label_dict(sample_filenames, label_filenames):
        N = len(sample_filenames)
        sample_label_dict = {}
        for i, sample_filename in enumerate(sample_filenames):
            label_filename = MapillaryDataset.get_first_filename_match(label_filenames, sample_filename)
            if label_filename is None:
                print("Missing label for")  # TODO: logger
            sample_label_dict[sample_filename] = label_filename
        return sample_label_dict
