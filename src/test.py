import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize

from config.configs import parse_test_config
from utils.helpers import torch_image_to_numpy
from data_processing.mapillary_dataset import MapillaryDataset
from data_processing.pipelines.processing_pipelines import (
                                                  create_data_preprocessing_pipeline,
                                                  create_prediction_postprocessing_pipeline
                                                  )
from models.segformer import create_segformer_model_for_inference


if __name__ == '__main__':
    # Config
    test_config = parse_test_config()

    img_shape = (test_config.img_height, test_config.img_width)

    # Data processing
    data_preprocessing_pipeline = create_data_preprocessing_pipeline(test_config)
    prediction_postprocessing_pipeline = create_prediction_postprocessing_pipeline(img_shape)

    # Dataset
    dataset = MapillaryDataset(test_config.test_data_path,
                               data_preprocessor=data_preprocessing_pipeline,
                               json_class_names_file_path=test_config.json_class_names_file_path)

    # Dataloader
    test_dataloader = DataLoader(dataset, batch_size=test_config.batch_size)

    # Class mappings
    id2name = deepcopy(dataset.id2name)
    id2name.pop(test_config.void_class_id)

    # Model
    model = create_segformer_model_for_inference(test_config, id2name)
    model.to(test_config.device)

    # Prediction loop
    for batch_num, batch in tqdm(enumerate(test_dataloader)):
        input_img = batch['pixel_values']
        raw_img = torch_image_to_numpy(resize(batch['raw_img'], img_shape).squeeze())
        input_img = input_img.to(test_config.device)

        outputs = model(input_img).logits
        predictions = prediction_postprocessing_pipeline(outputs, None)
        prediction_mask = dataset.apply_color_mask(predictions.squeeze())

        masked_img = cv2.addWeighted(raw_img, 0.5, prediction_mask, 0.5, 0)

        # Display prediction
        fig, ax = plt.subplots(2, figsize=(8, 8))
        ax[0].imshow(masked_img)
        ax[0].set_title('Prediction')
        ax[1].imshow(raw_img)
        ax[1].set_title('Ground truth')
        plt.show()

        if batch_num >= 0:
            break
