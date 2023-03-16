import sys
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

from data_processing.mapillary_dataset import MapillaryDataset
from data_processing.transformation_pipelines import (
                                                      create_evaluation_data_transformation_pipeline,
                                                      create_evaluation_label_transformation_pipeline
                                                      )
from data_processing.processing_pipelines import (
                                                  create_data_preprocessing_pipeline,
                                                  create_prediction_postprocessing_pipeline
                                                  )

sys.path.append('.')
sys.path.append('..')

from config.configs import parse_evaluation_config
from utils.metrics import Evaluator

    
if __name__ == '__main__':
    # Config
    evaluation_config = parse_evaluation_config()

    # Data transformations
    data_transformation_pipeline = create_evaluation_data_transformation_pipeline(evaluation_config)
    label_transformation_pipeline = create_evaluation_label_transformation_pipeline(evaluation_config)
    
    # Data processing
    data_preprocessing_pipeline = create_data_preprocessing_pipeline(evaluation_config)
    prediction_postprocessing_pipeline = create_prediction_postprocessing_pipeline(evaluation_config)

    # Dataset
    dataset = MapillaryDataset(evaluation_config.val_data_path,
                               evaluation_config.val_labels_path,
                               sample_transformation=data_transformation_pipeline,
                               label_transformation=label_transformation_pipeline,
                               data_preprocessor=data_preprocessing_pipeline,
                               json_class_names_file_path=evaluation_config.json_class_names_file_path)

    # Dataloader
    eval_dataloader = DataLoader(dataset, batch_size=evaluation_config.batch_size)

    # Model
    model = SegformerForSemanticSegmentation.from_pretrained(evaluation_config.model_checkpoint)
    model.to(evaluation_config.device)

    # Metrics
    evaluator = Evaluator(class_labels=dataset.class_color_dict.keys(),
                          ignore_classes=[evaluation_config.void_class_id])
    # iou_score = evaluate.load('mean_iou')

    # Evaluation loop
    for batch_num, batch in tqdm(enumerate(eval_dataloader)):
        img, label = batch
        img = img.to(evaluation_config.device)
        label = label.to(evaluation_config.device)
        
        outputs = model(img).logits
        predictions = prediction_postprocessing_pipeline(outputs)

        evaluator.update_state([predictions], [label])
        # results = iou_score.compute(predictions=[predictions], references=[label], num_labels=dataset.num_classes, ignore_index=evaluation_config.void_class_id)

        if batch_num >= 5:
            break

    print('----------------------')
    print('Mean IoU', evaluator.mean_iou())
    print('Class IoU', evaluator.mean_iou_per_class())
    print('----------------------')

    # print('Hugging face IoU')
    # print('Mean IoU', results['mean_iou'])
    # print('Class IoU', results['per_category_iou'])
