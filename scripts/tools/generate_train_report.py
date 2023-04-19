import sys
import os
import json
import matplotlib.pyplot as plt

sys.path.append('../..')
sys.path.append('../../src')

from pathlib import Path
from utils.helpers import load_training_state_log, plot_training_loss_with_iou, plot_class_distribution, plot_class_distribution_with_iou, plot_train_val_class_distribution, plot_train_val_class_distribution_with_iou
from data_processing.mapillary_dataset import MapillaryDataset


###########################################################################
### Model - adjust these params to match the model you want to evaluate ###
###########################################################################

MODEL_VERSION = 'model_B5_mapped'  # or model_B5
MODEL_CHECKPOINT = 94500

###########################################################################


# Paths
TRAINING_STATE_LOG_FILE = f'../../src/models/model_checkpoints/{MODEL_VERSION}/model_{MODEL_CHECKPOINT}/trainer_state.json'

# Dataset
VOID_CLASS_ID = 19
DATASET_DIR = '../../data/Mapillary_vistas_dataset'
TRAIN_LABEL_DISTRIBUTION_FILE = f'{DATASET_DIR}/mapped_train_class_distribution.json'
VAL_LABEL_DISTRIBUTION_FILE = f'{DATASET_DIR}/mapped_val_class_distribution.json'

TRAIN_DISTRIBUTION_PLOT_PATH = f'../../docs/images/mapped_class_distribution.png'
TRAIN_VAL_DISTRIBUTION_PLOT_PATH = f'../../docs/images/mapped_train_val_class_distribution.png'


# Output Paths
MODEL_REPORT_OUTPUT_DIR = f'../../docs/train_results/{MODEL_VERSION}/model_{MODEL_CHECKPOINT}'

LOSS_PLOT_PATH = f'{MODEL_REPORT_OUTPUT_DIR}/loss_history_with_iou_{MODEL_VERSION}_{MODEL_CHECKPOINT}.png'

IOU_TRAIN_DISTRIBUTION_PLOT_PATH = f'{MODEL_REPORT_OUTPUT_DIR}/train_class_distribution_with_iou_{MODEL_VERSION}_{MODEL_CHECKPOINT}.png'
IOU_VAL_DISTRIBUTION_PLOT_PATH = f'{MODEL_REPORT_OUTPUT_DIR}/val_class_distribution_with_iou_{MODEL_VERSION}_{MODEL_CHECKPOINT}.png'
TRAIN_VAL_IOU_DISTRIBUTION_PLOT_PATH = f'{MODEL_REPORT_OUTPUT_DIR}/train_val_class_distribution_with_iou_{MODEL_VERSION}_{MODEL_CHECKPOINT}.png'


def extract_train_log_data(training_log):
    train_loss_history = []
    val_loss_history = []
    eval_mean_iou_history = []

    for idx, logged_epoch in enumerate(training_log):
        if idx % 2 == 0:
            train_loss_history.append(logged_epoch['loss'])
        else:
            val_loss_history.append(logged_epoch['eval_loss'])
            eval_mean_iou_history.append(logged_epoch['eval_mean_iou'])
    return train_loss_history, val_loss_history, eval_mean_iou_history


if __name__ == '__main__':
    if not os.path.exists(MODEL_REPORT_OUTPUT_DIR):
        os.makedirs(MODEL_REPORT_OUTPUT_DIR)

    # Datasets
    train_dataset = MapillaryDataset(Path(f'{DATASET_DIR}/training/images'),
                                 Path(f'{DATASET_DIR}/training/labels_mapped'),
                                 json_class_names_file_path=Path('../../data/Mapillary_vistas_dataset/mapped_classes.json'))

    val_dataset = MapillaryDataset(Path(f'{DATASET_DIR}/validation/images'),
                                   Path(f'{DATASET_DIR}/validation/labels_mapped'),
                                   json_class_names_file_path=Path(f'{DATASET_DIR}/classes.json'))

    # Load the training state log
    training_log = load_training_state_log(TRAINING_STATE_LOG_FILE)

    train_loss_history, val_loss_history, eval_mean_iou_history = extract_train_log_data(training_log)

    # Validation class IoU
    mean_iou = eval_mean_iou_history[-1]
    class_iou = {int(k): v for k, v in sorted(training_log[-1]['eval_iou_per_class'].items(), key=lambda item: int(item[0]))}
    del class_iou[VOID_CLASS_ID]

    # Loss history
    if not os.path.exists(LOSS_PLOT_PATH):
        plot_training_loss_with_iou(train_loss_history, val_loss_history, eval_mean_iou_history)
        plt.savefig(LOSS_PLOT_PATH)
        plt.show()

    # Train class distribution
    if os.path.exists(TRAIN_LABEL_DISTRIBUTION_FILE):
        with open(TRAIN_LABEL_DISTRIBUTION_FILE, 'r', encoding='utf-8') as file:
            train_label_distribution = {int(k): v for k, v in sorted(json.load(file).items(), key=lambda item: int(item[0]))}
    else:
        train_label_distribution = train_dataset.get_label_distribution()
        with open(TRAIN_LABEL_DISTRIBUTION_FILE, 'w', encoding='utf-8') as file:
            json.dump(train_label_distribution, file, sort_keys=True)

    # Val class distribution
    if os.path.exists(VAL_LABEL_DISTRIBUTION_FILE):
        with open(VAL_LABEL_DISTRIBUTION_FILE, 'r', encoding='utf-8') as file:
            val_label_distribution = {int(k): v for k, v in sorted(json.load(file).items(), key=lambda item: int(item[0]))}
    else:
        val_label_distribution = val_dataset.get_label_distribution()
        with open(VAL_LABEL_DISTRIBUTION_FILE, 'w', encoding='utf-8') as file:
            json.dump(val_label_distribution, file, sort_keys=True)

    # Train dataset class distribution
    if not os.path.exists(TRAIN_DISTRIBUTION_PLOT_PATH):
        plot_class_distribution(train_label_distribution,
                                class_names=list(train_dataset.id2name.values()),
                                title='Class appearances in train set')
        plt.savefig(TRAIN_DISTRIBUTION_PLOT_PATH)
        plt.show()

    del train_label_distribution[VOID_CLASS_ID]
    del val_label_distribution[VOID_CLASS_ID]

    # Train class distribution with IoU
    if not os.path.exists(IOU_TRAIN_DISTRIBUTION_PLOT_PATH):
        plot_class_distribution_with_iou(train_label_distribution,
                                            class_iou,
                                            mean_iou,
                                            class_names=list(train_dataset.id2name.values()),
                                            title='Class appearances in train set with IoU')
        plt.savefig(IOU_TRAIN_DISTRIBUTION_PLOT_PATH)
        plt.show()


    # Val class distribution with IoU
    if not os.path.exists(IOU_VAL_DISTRIBUTION_PLOT_PATH):
        plot_class_distribution_with_iou(val_label_distribution,
                                         class_iou,
                                         mean_iou,
                                         class_names=list(train_dataset.id2name.values()),
                                         title='Class appearances in validation set with IoU')
        plt.savefig(IOU_VAL_DISTRIBUTION_PLOT_PATH)
        plt.show()


    # Train and val class distribution with val IoU
    if not os.path.exists(TRAIN_VAL_IOU_DISTRIBUTION_PLOT_PATH):
        plot_train_val_class_distribution_with_iou(train_label_distribution,
                                                   val_label_distribution, 
                                                   class_iou,
                                                   mean_iou,
                                                   class_names=list(train_dataset.id2name.values()),
                                                   title='Class appearances in train and validation set with IoU')
        plt.savefig(TRAIN_VAL_IOU_DISTRIBUTION_PLOT_PATH)
        plt.show()
