from pip import main
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from data_processing.mapillary_dataset import MapillaryDataset


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device


if __name__ == '__main__':
    data_path = Path('../data/Mapillary_vistas_dataset/training/images')  # load from ArgParse
    labels_path = Path('../data/Mapillary_vistas_dataset/training/labels')

    device = get_device()
    print(f'Available device: {device}')

    m_dataset = MapillaryDataset(data_path, labels_path)
    m_dataloader = DataLoader(m_dataset, batch_size=1, shuffle=True)

    img, label = m_dataset[0]
    
    m_dataset.display_image(img)
    m_dataset.display_image(label)

    for batch_num, batch in enumerate(m_dataloader):
        img, label = batch
        m_dataset.display_image(img[0])
        m_dataset.display_image(label[0])
        if batch_num >= 3:
            break
