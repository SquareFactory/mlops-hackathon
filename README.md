# MLOPs HAckathon
This repository contains all necessary examples for the MLOPS Hackathon.

## Goal
The goal of the hackathon, is to train and deploy a ML model using different platforms, while using multiple MLOPs tools to track and monitor the hackathon.

## Get started locally
To install the dependencies:
```
conda create -n mlopshackathon python=3.9
conda activate mlopshackathon
pip install -r requirements.txt
```

## Script usage
### Train a model
To train a model, execute the following script:
```bash
python training.py --max_epochs=5 --gpu=-1
```
### Inference
To perform basic inference, execute the following script:
```bash
python inference.py --checkpoint files/weights/MNIST_classifier_mobilenetv3_rwepoch=4-val_loss=0.04.ckpt --image files/imgs_inference/MNIST_digit.png
```