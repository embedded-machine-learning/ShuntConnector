# Shunt Connector

This repository implements the neural network compression technique called 'Shunt connection' using Keras and TensorFlow 2.x as its backend. Shunt connections were first introduced by [Singh et al.](https://www.researchgate.net/publication/334056710_Shunt_connection_An_intelligent_skipping_of_contiguous_blocks_for_optimizing_MobileNet-V2).

Shunt connections are applicable to any residual convolutional neural network architecture.

This repository was created as part of a master thesis, which includes more details on design choices and the limits of the implementation. It can be found under this [link](https://repositum.tuwien.at/bitstream/20.500.12708/17369/1/Haas%20Bernhard%20-%202021%20-%20Compressing%20MobileNet%20With%20Shunt%20Connections%20for%20NVIDIA...pdf).

Cite: 

`Bernhard Haas, Alexander Wendt, Axel Jantsch and Matthias Wess: Neural Network Compression Through Shunt Connections and Knowledge Distillation for Semantic Segmentation Problems, in Proceedings of Artificial Intelligence Applications and Innovations, 17th IFIP WG 12.5 International Conference, Greece (online), pp. 349-361, 2021, doi: https://doi.org/10.1007/978-3-030-79150-6`

and

`Bernhard Haas, Compressing MobileNet With Shunt Connections for NVIDIA Hardware, Master Thesis, TU Wien, 2021, url: https://publik.tuwien.ac.at/files/publik_295948.pdf`

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

Each step is controlled through the provided configuration file and can be replaced by a custom implementation.

### Logging

Each run produces a log file under ***log*** where summary of models and other results of the run are stored. The summary produced by the ***print_summary()*** call is also stored in this file.

Created models are saved under this folder in the .h5 format.

The path to the current logging folder is saved in ***shunt_connector.folder_name_logging***, such that it can be used in additional custom code.

## Installation

Run in root directory to install all dependencies:
`pip install -r requirements.txt`

If you want to install just the shunt connector package, run:
`pip install -e .`

### Basic Requirements

TensorFlow 2.X, tested for TensorFlow 2.3


## Examples

Examples show how this repository should be used.

For a "standard" experiment, use the ***standard_main.py*** under examples.
This main file executes the shunt connector procedure according to the provided configuration file.

For knowledge distillation, two scripts are available under examples. One in pure script format and another one in Jupyter form.

### Configuration File

All parameters for models, training and validation are set through a configuration file. A standard version of it can be produced by calling ***utils/create_config.py***. The produced file also holds information about possible values for nominal parameters.

### Custom Losses and Metrics

For semantic segmentation tasks, custom loss and metric have to be used. They are defined in ***utils/custom_loss_metrics.py***.

## Usage

A basic shunt insertion on the built in network architectures and datasets can be done by creating a new ***ShuntConnector*** object, initializing it with a ***configparser*** object and calling ***shunt_connector.execute()*** . This calls all 10 steps of the procedure serially. 

### Datasets

CIFAR10 and CIFAR100 experiments can be run out of the box, the datasets will be downloaded through Keras. Using cityscapes, requires preparing the dataset like in the official TensorFlow [repository](https://github.com/tensorflow/models/tree/master/research/deeplab/datasets).

Other datasets have to be loaded through custom code by replacing the ***create_dataset()*** call accordingly.

### Models

MobileNetV2 and MobileNetV3 are built-in in this repository, using slightly altered implementations from the keras_applications [repository](https://github.com/keras-team/keras-applications).

The sementic segmentation version introduced in the MobileNetV3 [paper](https://arxiv.org/abs/1905.02244) is also implemented in this repository.

## Custom Implementations

Each of the 10 steps can be replaced by a custom implementation. The methods are built in a way, that you can replace one of the steps but reuse all other ones. For example, it is possible to call ***create_dataset()*** and ***create_original_model()*** but then use a custom training procedure for training the original model. Afterwards you can still call ***create_shunt_model()*** and procede with other built-in methods. In order for this to work, certain variables have to be set at each step. If one variable is missing after calling a customized call, the program will terminate and an appropriate error message is shown.

Custom losses and metrics can be customized by setting the ***shunt_connector.task_losses*** and ***shunt_connector.task_metrics*** fields. Note that both fields have to hold **LISTS**, even when using only a single loss or metric. 

When defining custom models, losses or metrics, they have do be defined under Tensorflow's distributation scope. The scope is initialized when the ShuntConnector object is created and can be entered by calling ***shunt_connector.activate_distribution_scope()*** .

### Custom datasets

How to use custom datasets can be seen in the ***train_railsem*** example. It is necessary that the dataset_props dictionary gets properly initialized during the creation of the custom dataset. The field ***dataset_train*** and ***dataset_val*** hold tf.data objects and are used for training and validation accordingly. Note that these datasets have to be not 'batched' in this step, since the get 'batched' during the training or validation step using the batch size set in the corresponding config field.

### Custom models

Custom models can be used by calling ***set_..._model()*** instead of ***create_..._model()*** . The set model will be saved in the logging folder and FLOPs get calculated for the custom model. Note that this model does not have to be compiled, since it will be compiled before training and validation with the task specific tasks.

The example ***create_s_and_e_shunt_model.ipynb*** shows this process for using a custom shunt architecture.

### Custom training procedures

It is suggested to reuse as much code as possible when writing custom training procedures. How this can be done effficiently can be seen in ***train_final_ACE.py*** or ***train_final_dark_knowledge.ipynb*** in the examples folder, where the training of the final model was replaced by a knowledge distillation step.

## Additional Information

### Loading weights from a TensorFlow-Checkpoint file

Loading the weights of a TF1-Checkpoint file in Keras can be done by converting the tensors saved in the checkpoint files to numpy arrays and save them as .npy files. The script ***utils/convert_checkpoint_to_npy.py*** does this for the official TF-Slim model of the DeeplabV3-MobileNetV3 architecture, which can be found [here](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md).

