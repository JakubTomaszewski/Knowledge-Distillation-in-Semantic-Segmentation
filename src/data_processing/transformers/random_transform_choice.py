"""Module containing an object used for randomly picking data transformers

Raises:
    ValueError: raised when the number of transformers to pick is larger than the total number of transformers provided

Returns:
    np.ndarray: array containing the randomly chosen transformers
"""

from typing import List
import numpy as np


class RandomTransformChoice:
    """Class representing an object used for randomly picking data transformers.
    """
    def __init__(self,
                 transforms: List,
                 num_choices: int=1,
                 ) -> None:
        """Initializes the RandomTransformChoice. Sets the available transformers and how many of them will be picked.

        Args:
            transforms (List): list of transformers from which to pick
            num_choices (int): number of transformers to pick (min 0). Defaults to 1.

        Raises:
            ValueError: when the number of transformers to pick is larger than the total number of transformers provided
        """
        self.transforms = transforms
        if num_choices <= 0:
            num_choices = 1
        if num_choices > len(self.transforms):
            raise ValueError('Not enough transformers to choose from')
        self.num_choices = num_choices

    def __call__(self) -> np.ndarray:
        """Returns a array of randomly picked transformers.

        Returns:
            np.ndarray: array containing the randomly chosen transformers
        """
        return np.random.choice(self.transforms, size=self.num_choices, replace=False)
