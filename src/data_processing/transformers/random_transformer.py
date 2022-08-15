"""Module containing an object used for randomly choosing and applying data transformations

Raises:
    ValueError: raised when the number of transforms to pick is larger than the total number of transforms provided

Returns:
    torch.Tensor: transformed sample
"""

from typing import List
import numpy as np
import torch


class RandomTransformer:
    """Class representing an object used for randomly choosing and applying data transformations.
    """
    def __init__(self,
                 transforms: List,
                 num_choices: int=1,
                 ) -> None:
        """Initializes the RandomTransformer class. Sets the available transforms and how many of them will be picked.

        Args:
            transforms (List): list of transforms from which to pick
            num_choices (int): number of transforms to pick (min 0). Defaults to 1.

        Raises:
            ValueError: when the number of transforms to pick is larger than the total number of transforms provided
        """
        self.transforms = transforms
        if num_choices < 0:
            raise ValueError('Number of transforms to apply cannot be less than 0')
        if num_choices > len(self.transforms):
            raise ValueError('Not enough transforms to choose from')
        self.num_choices = num_choices

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        """Applies randomly chosen transforms to a given sample.

        Args:
            sample (torch.Tensor): sample to be transformed

        Returns:
            torch.Tensor: transformed sample
        """
        random_transforms = np.random.choice(self.transforms, size=self.num_choices, replace=False)

        for transform in random_transforms:
            sample = transform(sample)
        return sample
