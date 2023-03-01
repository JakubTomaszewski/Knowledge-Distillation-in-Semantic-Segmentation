import sys
from torch.utils.data import DataLoader
from utils.metrics import Evaluator

from data_processing.mapillary_dataset import MapillaryDataset
from data_processing.data_transformations import (
                                                 create_data_transformation_pipeline,
                                                 create_label_transformation_pipeline
                                                 )

sys.path.append('.')

from utils.helpers import set_randomness_seed, get_device, display_dict
from config.dataset_config import parse_dataset_config


if __name__ == '__main__':
    dataset_config = parse_dataset_config()

    set_randomness_seed(dataset_config.seed)
    device = get_device()
    print(f'Available device: {device}')

    data_transformation_pipeline = create_data_transformation_pipeline(dataset_config)
    label_transformation_pipeline = create_label_transformation_pipeline(dataset_config)

    m_dataset = MapillaryDataset(dataset_config.train_data_path,
                                 dataset_config.train_labels_path,
                                 sample_transformation=data_transformation_pipeline,
                                 label_transformation=label_transformation_pipeline,
                                 json_class_names_file_path=dataset_config.json_class_names_file_path)

    m_dataloader = DataLoader(m_dataset, batch_size=dataset_config.batch_size, shuffle=True)

    evaluator = Evaluator(m_dataset.class_color_dict.keys())

    # img, label = m_dataset[50]
    
    # m_dataset.display_torch_image(img)
    # m_dataset.display_torch_image(label)
    # masked_label = m_dataset.apply_color_mask(label.squeeze().numpy())
    # m_dataset.display_numpy_image(masked_label)

    for batch_num, batch in enumerate(m_dataloader):
        img, label = batch
        
        evaluator.update_state(label, label)
        # mean_iou_per_class = evaluate_predictions(label, label)
        # display_dict(mean_iou_per_class)
        
        # m_dataset.display_torch_image(img[0])
        # m_dataset.display_torch_image(label[0])
        # masked_label = m_dataset.apply_color_mask(label[0].squeeze().numpy())
        # m_dataset.display_numpy_image(masked_label)
        
        if batch_num >= 1:
            break
    
    print(evaluator.mean_iou())
    print(evaluator.mean_iou_per_class())
    evaluator.reset_state()
    print(evaluator.mean_iou())
    print(evaluator.mean_iou_per_class())
    
