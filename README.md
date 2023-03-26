# SegFormer Semantic Segmentation with Knowledge Distillation

## Overview

This project is being created for my Bachelor's Thesis. 
Its main goal is to create a small and robust Semantic Segmentation model. Eventually, the model will be used for segmenting road scenes in a fully autonomous vehicle. Thus, while preserving good performance the model has to work in real time.


To achive both, a technique known as Knowledge Distillation will be used in order to transfer knowledge from a large and complex model, to a small one.


## Technologies

![Python 3.8](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=blue)

![numpy](https://img.shields.io/badge/Numpy-1.22.4-777BB4?style=for-the-badge&logo=numpy&logoColor=white)

![PyTorch](https://img.shields.io/badge/PyTorch-1.12.0-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)

- [HuggingFace 🤗 Transformers](https://huggingface.co/docs/transformers/index)

- [MLFlow](https://mlflow.org/docs/latest/index.html)




## Creating the development environment

```sh
$ conda create --name <env> --file requirements_conda.txt python=3.8
```

or

```sh
$ conda create --name <env> python=3.8
$ pip install -r requirements.txt
```
