import sys
import matplotlib.pyplot as plt

sys.path.append('../..')
sys.path.append('../../src')

from pathlib import Path
from utils.helpers import load_training_state_log, plot_training_loss_with_iou, plot_class_distribution_with_iou
from data_processing.mapillary_dataset import MapillaryDataset


TRAINING_STATE_LOG_FILE_1 = '../../src/models/model_checkpoints/model_99000/trainer_state.json'
TRAINING_STATE_LOG_FILE_2 = '../../src/models/model_checkpoints/model_202500/trainer_state.json'


if __name__ == '__main__':
    # Load the training state log
    training_state_log = load_training_state_log(TRAINING_STATE_LOG_FILE_1)
    
    train_loss = []
    val_loss = []
    eval_mean_iou = []
    
    for idx, logged_epoch in enumerate(training_state_log):
        if idx % 2 == 0:
            train_loss.append(logged_epoch['loss'])
        else:
            val_loss.append(logged_epoch['eval_loss'])
            eval_mean_iou.append(logged_epoch['eval_mean_iou'])

    plot_training_loss_with_iou(train_loss, val_loss, eval_mean_iou)
    plt.show()


    class_iou = training_state_log[-1]['eval_iou_per_class']

    # Dataset
    m_dataset = MapillaryDataset(Path('../../data/Mapillary_vistas_dataset/training/images'),
                                 Path('../../data/Mapillary_vistas_dataset/training/labels'),
                                 json_class_names_file_path=Path('../../data/Mapillary_vistas_dataset/classes.json'))

    label_distribution = m_dataset.get_label_distribution()
    plot_class_distribution_with_iou(label_distribution, class_iou, class_names=list(m_dataset.id2name.values()))
    plt.savefig('../../docs/class_distribution_with_iou.png')
    plt.show()

