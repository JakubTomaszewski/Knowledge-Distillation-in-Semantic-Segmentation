"""Helper functions
"""

import random
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List


def torch_image_to_numpy(image: torch.Tensor) -> np.ndarray:
    """Converts a torch.Tensor image to a numpy array.

    Args:
        image (torch.Tensor): image to convert

    Returns:
        np.ndarray: converted image
    """
    return image.permute(1, 2, 0).numpy()


def set_randomness_seed(seed):
    """Sets a seed for computations performed by torch and random library.

    Args:
        seed (int): random seed to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_available_devices() -> list:
    """Returns all available devices for computation.

    Returns:
        list: available device for computation
    """
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    if torch.backends.mps.is_available():
        devices.append('mps')
    print('Available devices:', devices)
    return devices


def available_torch_device(device):
    """Returns the device available for computation.

    Returns:
        torch.device: device available for computation
    """
    if device in get_available_devices():
        print(f'Chosen device: {device}')
        return torch.device(device)
    else:
        return torch.device('cpu')


def display_dict(dict_to_print: Dict):
    """Displays a dict in a fancy way

    Args:
        dict_to_print (_type_): _description_
    """
    print("\n".join("{}:  {}".format(k, v) for k, v in dict_to_print.items()))


def is_image(path: str) -> bool:
    """Checks if a given path is an image.

    Args:
        path (str): path to check

    Returns:
        bool: True if the path is an image, False otherwise
    """
    return path.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))


def load_training_state_log(json_file_path: str) -> dict:
    """Loads the training state log from a json file.

    Args:
        json_file_path (str): path to the json file

    Returns:
        dict: training state log
    """
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        return json.load(json_file)['log_history']



# ------------ Plotting functions ------------ #


def plot_training_loss(train_loss, val_loss=None):
    """Plots the training loss. May also plot the validation loss.

    Args:
        training_loss (list): list of training loss
        
        val_loss (list, optional): list of validation loss. Defaults to None.
    """
    fig = plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label='train')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    if val_loss is not None:
        plt.plot(val_loss, label='val')
        plt.legend()
    
    return fig


def plot_training_loss_with_iou(train_loss, val_loss, eval_mean_iou):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_loss, label='train')
    ax.set_title('Training loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.plot(val_loss, label='val')

    ax2 = ax.twinx()
    ax2.plot(eval_mean_iou, label='eval_mean_iou', color='green', linestyle='--', linewidth=2)
    ax2.set_ylabel('Val Mean IoU')

    fig.legend(loc='center right', bbox_to_anchor=(0.9, 0.5), ncol=1)
    return fig


def plot_class_iou(class_iou: Dict,
                   title: str = None,
                   class_names: List = None,
                   mean_iou: float = None
                   ) -> plt.Figure:
    # Creates a bar plot with the class IoU
    
    fig = plt.figure(figsize=(8, 5))
    plt.bar(class_iou.keys(), class_iou.values(), color='green', alpha=0.5)
    plt.ylim(0, 1.1)
    plt.ylabel('IoU')
    plt.xlabel('Class')

    if title is not None:
        plt.title(title)
    else:
        plt.title('IoU score for each class')
    
    if class_names is not None:
        plt.xticks(range(len(class_iou.keys())), class_names, rotation=90)
        plt.tight_layout()
    
    if mean_iou is not None:
        plt.title(f'Mean IoU: {mean_iou}', loc='left', fontweight='semibold', fontsize = 9, color='red')
    
    return fig


def plot_class_distribution(class_counts: dict,
                            class_names: list=None,
                            title: str=None
                            ) -> plt.Figure:
    """Creates a bar plot with the class distribution

    Args:
        class_counts (dict): dict with class counts
    """
    fig = plt.figure(figsize=(15, 7))
    plt.bar(class_counts.keys(), class_counts.values(), color='green', alpha=0.5)
    plt.ylabel('Count')
    plt.xlabel('Class')
    
    if title is not None:
        plt.title(title)
    else:
        plt.title('Class appearances')

    if class_names is not None:
        plt.xticks(list(class_counts.keys()), class_names, rotation=90)
        plt.tight_layout()
    return fig


def plot_train_val_class_distribution(train_class_counts: Dict,
                                     val_class_counts: Dict,
                                     class_names: List=None,
                                     title: str=None
                                     ) -> plt.Figure:
    """Creates a bar plot with the class distribution

    Args:
        class_counts (dict): dict with class counts
    """
    train_class_counts_keys = np.array(list(train_class_counts.keys()), dtype=np.int8)
    val_class_counts_keys = np.array(list(val_class_counts.keys()), dtype=np.int8)
    
    fig, ax = plt.subplots(figsize=(17, 7))
    ax.bar(train_class_counts_keys - 0.2, train_class_counts.values(), color='green', alpha=0.5, width=0.4, align='edge')
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Train Count', color='green')
    
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('Class appearances')

    ax2 = ax.twinx()
    ax2.bar(val_class_counts_keys + 0.2, val_class_counts.values(), color='blue', alpha=0.5, width=0.4, align='edge')
    ax2.set_ylabel('Val Count', color='blue')

    if class_names is not None:
        present_class_names = [class_names[int(class_id)] for class_id in train_class_counts.keys()]
        ax.set_xticks(train_class_counts_keys, present_class_names, rotation=90)
        fig.tight_layout()
    return fig


def plot_train_val_class_distribution_with_iou(train_class_counts: Dict,
                                               val_class_counts: Dict,
                                               class_iou: Dict,
                                                mean_iou: int=None,
                                               class_names: List=None,
                                               title: str=None
                                               ) -> plt.Figure:
    """Creates a bar plot with the class distribution

    Args:
        class_counts (dict): dict with class counts
    """
    train_class_counts_keys = np.array(list(train_class_counts.keys()), dtype=np.int8)
    val_class_counts_keys = np.array(list(val_class_counts.keys()), dtype=np.int8)
    
    fig, ax = plt.subplots(figsize=(17, 7))
    ax.set_title('Train and Val class distribution with IoU')
    ax.bar(train_class_counts_keys - 0.2, train_class_counts.values(), color='green', alpha=0.5, width=0.4)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Train Count', color='green')
    
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('Class appearances')

    ax2 = ax.twinx()
    ax2.bar(val_class_counts_keys + 0.2, val_class_counts.values(), color='blue', alpha=0.5, width=0.4)
    ax2.set_ylabel('Val Count', color='blue')

    ax3 = ax.twinx()
    ax3.set_ylim(0, 1.1)
    ax3.plot(class_iou.keys(), class_iou.values(), color='red', linestyle='--', marker='o', linewidth=2)
    ax3.set_yticks([])
    
    plt.legend(['IoU'])

    if mean_iou is not None:
        plt.text(0.01, 0.96, f'Mean IoU: {mean_iou}', transform=ax.transAxes, fontweight='semibold', fontsize = 11, color='red')

    if class_names is not None:
        present_class_names = [class_names[int(class_id)] for class_id in train_class_counts.keys()]
        ax.set_xticks(train_class_counts_keys, present_class_names, rotation=90)
        fig.tight_layout()
    return fig


def plot_class_distribution_with_iou(class_counts: Dict, 
                                     class_iou: Dict,
                                     mean_iou: int=None,
                                     class_names: List=None,
                                     title: str=None
                                     ) -> plt.Figure:
    """Creates a bar plot with the class distribution and with an IoU score for each class.

    Args:
        class_counts (dict): _description_
        class_iou (dict): _description_
        class_names (list, optional): _description_. Defaults to None.

    Returns:
        plt.Figure: _description_
    """
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.bar(class_counts.keys(), class_counts.values(), color='green', alpha=0.5)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('Class appearances')

    ax2 = ax.twinx()
    ax2.set_ylim(0, 1.1)
    ax2.plot(class_iou.keys(), class_iou.values(), color='red', linestyle='--', marker='o', linewidth=2)
    ax2.set_ylabel('IoU')

    if mean_iou is not None:
        plt.text(0.01, 0.96, f'Mean IoU: {mean_iou}', transform=ax.transAxes, fontweight='semibold', fontsize = 11, color='red')

    if class_names is not None:
        present_class_names = [class_names[int(class_id)] for class_id in class_counts.keys()]
        ax.set_xticks(list(class_counts.keys()), present_class_names, rotation=90)
        fig.tight_layout()
    return fig
