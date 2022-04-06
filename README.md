# MLOPs Hackathon
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
These scripts should be enough to try out various ML platforms, be it for training or deployment.

## Hackathon
The hackathon is split into three parts: training platforms, inference platforms and MLOPs solutions. These should be apporached in parallel, with multiple teams.

### Training team
The training team should take the training code and perform trainings on different platforms. Notes should be taken on how these platforms approach training, and what they offer for our use case. The training times, prices and performances should be benchmarked as well. 
Platforms to try out:
- Grid.ai
- Vertex ai
- 

### Inference team
The inference team should take the inference script with the provided weights and deploy it to different platforms. In parallel, they should develop a script to test the platforms at different loads (1-10000 simultaneous inferences). Notes should be taken on the facility to deploy, as well as the offering of services for deployment.
- Vertex ai
- ...

### MLOPs team
The MLOPs team should test the integration of different frameworks into the code, capturing the correct metrics. The ideal scenario would be that all the trainings and inferences performed by the two respective teams can be centrally monitored, and the results be compared accross multiple platforms.
Frameworks to test:
- MLFlow
- Kubeflow
- clearml
