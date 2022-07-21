from pip import main
from pathlib import Path
from mapillary_dataset import MapillaryDataset


if __name__ == '__main__':
    data_path = Path('../data/Mapillary_vistas_dataset/training/images')  # load from ArgParse
    labels_path = Path('../data/Mapillary_vistas_dataset/training/labels')

    m_dataset = MapillaryDataset(data_path, labels_path)
    img, label = m_dataset[0]
    m_dataset.display_image(img)
    m_dataset.display_image(label)
