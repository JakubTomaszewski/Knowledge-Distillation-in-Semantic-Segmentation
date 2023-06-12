# Based on https://deci.ai/blog/measure-inference-time-deep-neural-networks/

import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader

from config.configs import parse_test_config
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
    model = create_segformer_model_for_inference(test_config.model_checkpoint, id2name)
    model.to(test_config.device)

    
    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    num_repetitions = 300
    timings = np.zeros((num_repetitions, 1))
    
    #GPU-WARM-UP
    dummy_input = torch.randn(1, 3, 512, 1024, dtype=torch.float).to(test_config.device)
    for _ in range(10):
        _ = model(dummy_input)

    # Prediction loop
    for batch_num, batch in tqdm(enumerate(test_dataloader)):
        input_img = batch['pixel_values']
        input_img = input_img.to(test_config.device)

        starter.record()
        _ = model(input_img)
        ender.record()
        
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[batch_num] = curr_time

        if batch_num >= num_repetitions - 1:
            break

    mean_syn = np.sum(timings) / num_repetitions
    std_syn = np.std(timings)
    
    print('-----------------------------')
    print(f'AVG INFERENCE TIME: {mean_syn}')
    print('--------------------------------')
