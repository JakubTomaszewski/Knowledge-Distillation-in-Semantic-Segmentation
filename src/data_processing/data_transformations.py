import torch
from torchvision.transforms import ToTensor, RandomRotation, RandomCrop
from torchvision.transforms import Compose

def create_data_transformation_pipeline(rotation_angle) -> Compose:
    return Compose([# Resize
                    RandomRotation(rotation_angle, fill=0),
                    # RandomCrop(0.9)
                    ])


def create_label_transformation_pipeline(rotation_angle) -> Compose:
    return Compose([RandomRotation(rotation_angle, fill=0),
                    ])
