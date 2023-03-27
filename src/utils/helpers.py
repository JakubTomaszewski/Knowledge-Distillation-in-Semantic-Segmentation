"""Helper functions
"""

import random
import json
import torch
import numpy as np
import matplotlib.pyplot as plt


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


def display_dict(dict_to_print):
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
    fig = plt.figure()
    plt.plot(train_loss, label='train')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    if val_loss is not None:
        plt.plot(val_loss, label='val')
        plt.legend()
    return fig


def plot_training_loss_with_iou(train_loss, val_loss, eval_mean_iou):
    fig, ax = plt.subplots()
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


def plot_class_distribution(class_counts: dict, class_names: list=None) -> plt.Figure:
    """Creates a bar plot with the class distribution

    Args:
        class_counts (dict): dict with class counts
    """
    fig = plt.figure(figsize=(15, 7))
    plt.bar(class_counts.keys(), class_counts.values(), color='green', alpha=0.5)
    plt.ylabel('Count')
    plt.xlabel('Class')
    plt.title('Class appearances')

    if class_names is not None:
        present_class_names = [class_names[int(class_id)] for class_id in class_counts.keys()]
        plt.xticks(list(class_counts.keys()), present_class_names, rotation=90)
        plt.tight_layout()
    return fig


def plot_class_distribution_with_iou(class_counts: dict, class_iou: dict, class_names: list=None) -> plt.Figure:
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
    ax.set_title('Class appearances')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')

    ax2 = ax.twinx()
    ax2.plot(class_iou.keys(), class_iou.values(), color='red', linestyle='--', marker='o', linewidth=2)
    ax2.set_ylabel('IoU')

    if class_names is not None:
        present_class_names = [class_names[int(class_id)] for class_id in class_counts.keys()]
        ax.set_xticks(list(class_counts.keys()), present_class_names, rotation=90)
        fig.tight_layout()
    return fig
