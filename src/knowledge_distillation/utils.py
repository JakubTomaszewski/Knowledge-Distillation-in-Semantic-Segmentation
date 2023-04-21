import torch
import torch.nn as nn
from typing import Tuple


def resize_outputs(output: torch.Tensor,
                   output_size: Tuple[int, int],
                   mode: str = 'bilinear',
                   ) -> torch.Tensor:
    """Resizes the outputs to a given size to match the labels.

    Args:
        output (torch.Tensor): model output tensor to be resized
        output_size (Tuple[int, int]): desired size

    Returns:
        torch.Tensor: resized output tensor
    """
    output_resized = nn.functional.interpolate(
                            output,
                            size=output_size, # (height, width)
                            mode=mode,
                            align_corners=False
                            )
    return output_resized
