# SegFormer Semantic Segmentation with Knowledge Distillation

## Overview

This project is being created for my Bachelor's Thesis - _"Tuning Small Semantic Segmentation Models via Knowledge Distillation"_.
The idea was to create a small and robust Semantic Segmentation model, for the purpose of segmenting road scenes in a fully autonomous vehicle. Hence, while preserving good performance the model has to work in real time.

To achive both, a technique known as Knowledge Distillation is employed in order to transfer knowledge from a large and complex [SegFormer](https://arxiv.org/abs/2105.15203) B5 model, to a small SegFormer B0 model, being lightweight enough to work on most devices in real time.

<p align="center">
  <img width=80% src="./docs/images/visualization.gif" />
</p>

## Technologies

![Python 3.8](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=blue)

![numpy](https://img.shields.io/badge/Numpy-1.22.4-777BB4?style=for-the-badge&logo=numpy&logoColor=white)

![PyTorch](https://img.shields.io/badge/PyTorch-1.12.0-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)

- [HuggingFace 🤗 Transformers](https://huggingface.co/docs/transformers/index)

- [MLFlow](https://mlflow.org/docs/latest/index.html)

## Method

Response-Based Knowledge Distillation in employed while training the model. The training pipeline is presented in the diagram below. See [knowledge_distillation](./src/knowledge_distillation) for the implementation details.

<p align="center">
  <img width=80% src="./docs/diagrams/rb_kd.png" />
</p>

## Creating the development environment

```sh
$ conda create --name <env> --file requirements_conda.txt python=3.8
```

or

```sh
$ conda create --name <env> python=3.8
$ pip install -r requirements.txt
```

## Train

```sh
$ python src/train.py
```

## Visualize training logs

```sh
mlflow ui --backend-store-uri src/models/mlflow_logs
```
