import sys
from torch.utils.data import DataLoader
from utils.metrics import Evaluator

from data_processing.mapillary_dataset import MapillaryDataset
from data_processing.transformation_pipelines import (
                                                 create_data_transformation_pipeline,
                                                 create_label_transformation_pipeline
                                                 )
from data_processing.processing_pipelines import (
                                                create_data_preprocessing_pipeline,
                                                create_prediction_postprocessing_pipeline
                                                )



sys.path.append('.')

from utils.helpers import set_randomness_seed, display_dict
from config.configs import parse_train_config


if __name__ == '__main__':
    # Config
    train_config = parse_train_config()

    set_randomness_seed(train_config.seed)

    # Data transformations
    data_transformation_pipeline = create_data_transformation_pipeline(train_config)
    label_transformation_pipeline = create_label_transformation_pipeline(train_config)
    
    # Data processing
    data_preprocessing_pipeline = create_data_preprocessing_pipeline(train_config)
    prediction_postprocessing_pipeline = create_prediction_postprocessing_pipeline(train_config)
    
    # Dataset
    m_dataset = MapillaryDataset(train_config.train_data_path,
                               train_config.train_labels_path,
                               sample_transformation=data_transformation_pipeline,
                               label_transformation=label_transformation_pipeline,
                               data_preprocessor=data_preprocessing_pipeline,
                               json_class_names_file_path=train_config.json_class_names_file_path)

    # Dataloader
    m_dataloader = DataLoader(m_dataset, batch_size=train_config.batch_size, shuffle=True)

    # # Model
    # model = SegformerForSemanticSegmentation.from_pretrained(evaluation_config.model_checkpoint)
    # model.to(evaluation_config.device)

    # Metrics
    evaluator = Evaluator(class_labels=m_dataset.class_color_dict.keys(),
                          ignore_classes=[train_config.void_class_id])

    for batch_num, batch in enumerate(m_dataloader):
        img, label = batch
        img = img.to(train_config.device)
        label = label.to(train_config.device)

        evaluator.update_state(label, label)

        # m_dataset.display_torch_image(img[0].cpu())
        # m_dataset.display_torch_image(label[0])
        # masked_label = m_dataset.apply_color_mask(label[0].cpu().squeeze().numpy())
        # m_dataset.display_numpy_image(masked_label)
        
        if batch_num >= 2:
            break
    
    print(evaluator.mean_iou())
    print(evaluator.mean_iou_per_class())
    
