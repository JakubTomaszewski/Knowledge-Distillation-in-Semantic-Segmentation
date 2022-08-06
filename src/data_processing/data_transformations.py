import torch
from torchvision.transforms import ToTensor, RandomRotation, RandomCrop
from torchvision.transforms import Compose

def create_transformation_pipeline() -> Compose:
    return Compose([RandomRotation(20, expand=False, fill=0),
                    # RandomCrop(0.9)
                    ])
