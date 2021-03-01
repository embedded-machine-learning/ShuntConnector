# Shunt Connector

## Description

The shunt connector procedure consits of 10 steps:

1. Create dataset
2. Create original model
3. Train original model
4. Calculate knowledge quotients
5. Create shunt model
6. Train shunt model
7. Create final model
8. Train final model
9. Test latency
10. Print summary

Each step is controlled through the provided configuration file.

Each run produces a log file under ***log***. Created models are also saved under this folder.

CIFAR10 and CIFAR100 experiments can be run out of the box. Using cityscapes, requires preparing the dataset like in the official tensorflow [repository](https://github.com/tensorflow/models/tree/master/research/deeplab/datasets).

## Installation

Run in root directory:
`pip install -r requirements.txt`

## Usage

Examples show how this repository should be used.

For a "standard" experiment, use the ***standard_main.py*** under examples.
This main file executes the shunt connector procedure according to the provided configuration file.

For knowledge distillation, two other scripts are available under examples. One in pure script format and another one in Jupyter form.

### Configuration File

All parameters for models, training and validation are set through a configuration file. A standard version of it can be produced by calling ***utils/create_config.py***. The produced file also holds information about possible values for nominal parameters.

### Custom Losses and Metrics

For semantic segmentation tasks, custom loss and metric have to be used. They are defined in ***utils/custom_loss_metrics.py***.