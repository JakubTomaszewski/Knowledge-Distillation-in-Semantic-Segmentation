# import evaluate
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader

from config.configs import parse_evaluation_config
from utils.helpers import display_dict, plot_class_iou
from utils.metrics import Evaluator
from data_processing.mapillary_dataset import MapillaryDataset
from data_processing.pipelines.transformation_pipelines import (
                                                      create_evaluation_data_transformation_pipeline,
                                                      create_evaluation_label_transformation_pipeline
                                                      )
from data_processing.pipelines.processing_pipelines import (
                                                  create_data_preprocessing_pipeline,
                                                  create_prediction_postprocessing_pipeline
                                                  )
from models.segformer import create_segformer_model_for_inference


if __name__ == '__main__':
    # Config
    evaluation_config = parse_evaluation_config()

    img_shape = (evaluation_config.img_height, evaluation_config.img_width)

    # Data transformations
    data_transformation_pipeline = create_evaluation_data_transformation_pipeline(evaluation_config)
    label_transformation_pipeline = create_evaluation_label_transformation_pipeline(evaluation_config)

    # Data processing
    data_preprocessing_pipeline = create_data_preprocessing_pipeline(evaluation_config)
    prediction_postprocessing_pipeline = create_prediction_postprocessing_pipeline(img_shape)

    # Dataset
    dataset = MapillaryDataset(evaluation_config.val_data_path,
                               evaluation_config.val_labels_path,
                            #    sample_transformation=data_transformation_pipeline,
                            #    label_transformation=label_transformation_pipeline,
                               data_preprocessor=data_preprocessing_pipeline,
                               json_class_names_file_path=evaluation_config.json_class_names_file_path)

    # Dataloader
    eval_dataloader = DataLoader(dataset, batch_size=evaluation_config.batch_size)

    # Class mappings
    id2name = deepcopy(dataset.id2name)
    id2name.pop(evaluation_config.void_class_id)

    # Model
    model = create_segformer_model_for_inference(evaluation_config, id2name)
    model.to(evaluation_config.device)

    # Metrics
    evaluator = Evaluator(class_labels=dataset.id2name.keys(),
                          ignore_index=evaluation_config.void_class_id)
    # iou_score = evaluate.load('mean_iou')

    # Evaluation loop
    for batch_num, batch in tqdm(enumerate(eval_dataloader)):
        img, label = batch['pixel_values'], batch['labels']
        img = img.to(evaluation_config.device)
        label = label.to(evaluation_config.device)

        outputs = model(img).logits
        predictions = prediction_postprocessing_pipeline(outputs, None)

        evaluator.update_state([predictions.cpu().numpy()], [label.cpu().numpy()])
        # iou_score.add_batch(predictions=predictions, references=label)


    # IoU Calculation
    mean_iou = evaluator.internal_state_mean_iou()
    class_iou = evaluator.internal_state_iou_score_per_class()

    print('----------------------')
    print(f'Mean IoU: {mean_iou}')
    print('Class IoU:')
    display_dict(class_iou)
    print('----------------------')

    # Extract classes present in the validation dataset
    present_class_iou = {k: v for k, v in zip(id2name.values(), class_iou.values()) if v is not None}

    # Plot class IoU
    plot_class_iou(present_class_iou,
                   title='Class IoU score on the validation dataset',
                   class_names=present_class_iou.keys(),
                   mean_iou=mean_iou)
    plt.show()


    # print('Hugging face IoU')
    # results = iou_score.compute(num_labels=dataset.num_classes, ignore_index=evaluation_config.void_class_id)
    # print('Mean IoU', results['mean_iou'])
    # print('Class IoU', results['per_category_iou'])
