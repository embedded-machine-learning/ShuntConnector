# -*- coding: utf-8 -*-
"""
Step #1 of the shunt connection procedure.

Copyright 2021 Christian Doppler Laboratory for Embedded Machine Learning

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


# Built-in/Generic Imports
from pathlib import Path

# Libs
import tensorflow as tf
import tensorflow_datasets as tfds

# Own modules
from shunt_connector.utils import custom_generators

__author__ = 'Bernhard Haas'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'Apache 2.0'
__version__ = '1.0.0'
__maintainer__ = 'Bernhard Haas'
__email__ = 'bernhardhaas55@gmail.com'
__status__ = 'Release'

def create_dataset(self):
    """This method represents step #1 of the shunt connection procedure.
       It creates a tf.data object for training and validation and sets
       the values for the dataset properties dictionary

    Raises:
        Exception: Raises an exception when configuration file does not hold information for the dataset
    """

    print('\nCreate dataset')

    if not self.dataset_params:
        raise Exception('No parameters found in config for creating the dataset! Create the field [DATASET]')

    if self.dataset_params['name'] == 'CIFAR10':

        self.dataset_props['num_classes'] = 10
        self.dataset_props['input_shape'] = (32,32,3)
        self.dataset_props['len_train_data'] = 40000
        self.dataset_props['len_val_data'] = 10000
        self.dataset_props['len_test_data'] = 10000
        self.dataset_props['task'] = 'classification'
        self.test_batchsize = self.dataset_params['test_batchsize']

        self.dataset_train = custom_generators.create_CIFAR_dataset(num_classes=10,
                                                                    dataset_type='train')
        self.dataset_val = custom_generators.create_CIFAR_dataset(num_classes=10,
                                                                  dataset_type='val')
        self.dataset_test = custom_generators.create_CIFAR_dataset(num_classes=10,
                                                                   dataset_type='test')
        print('CIFAR10 was loaded successfully!')

    if self.dataset_params['name'] == 'CIFAR100':

        self.dataset_props['num_classes'] = 100
        self.dataset_props['input_shape'] = (32,32,3)
        self.dataset_props['len_train_data'] = 40000
        self.dataset_props['len_val_data'] = 10000
        self.dataset_props['len_test_data'] = 10000
        self.dataset_props['task'] = 'classification'
        self.test_batchsize = self.dataset_params['test_batchsize']

        self.dataset_train = custom_generators.create_CIFAR_dataset(num_classes=100,
                                                                    dataset_type='train')
        self.dataset_val = custom_generators.create_CIFAR_dataset(num_classes=100,
                                                                  dataset_type='val')
        self.dataset_test = custom_generators.create_CIFAR_dataset(num_classes=100,
                                                                   dataset_type='test')
        print('CIFAR100 was loaded successfully!')

    if self.dataset_params['name'] == 'cityscapes':

        self.dataset_props['num_classes'] = 19
        self.dataset_props['input_shape'] = (self.dataset_params['input_size'][0],
                                             self.dataset_params['input_size'][1],
                                             3)
        self.dataset_props['len_train_data'] = 2975
        self.dataset_props['len_test_data'] = 500   # test = val
        self.dataset_props['len_val_data'] = 500   
        self.dataset_props['task'] = 'segmentation'
        self.test_batchsize = self.dataset_params['test_batchsize']

        self.dataset_train = custom_generators.create_cityscape_dataset(Path(self.dataset_params['path']),
                                                                             self.dataset_props['input_shape'], is_training=True)
        self.dataset_val = custom_generators.create_cityscape_dataset(Path(self.dataset_params['path']),
                                                                           self.dataset_props['input_shape'], is_training=False)
        self.dataset_test = self.dataset_val
        print('Successfully loaded cityscapes dataset with input shape: {}'.format(self.dataset_props['input_shape']))

    if self.dataset_params['name'] == 'WIDER_FACE':

        tfds.list_builders()
        builder = tfds.builder('wider_face')
        builder.download_and_prepare()
        self.dataset_train = builder.as_dataset(split='train', shuffle_files=True)

    if self.dataset_params['name'] == 'MNIST_Objects':

        assert self.dataset_params['input_size'][0] == self.dataset_params['input_size'][1]

        layer_widths = [28,14,7,4,2,1]
        num_boxes = [3,3,3,3,3,3]

        self.dataset_train, self.dataset_test = custom_generators.create_MNIST_Objects_dataset(self.dataset_params['input_size'][0], layer_widths=layer_widths, num_boxes=num_boxes)
        self.dataset_val = self.dataset_test

        self.dataset_props['num_classes'] = 10
        self.dataset_props['input_shape'] = (self.dataset_params['input_size'][0], self.dataset_params['input_size'][1],3)
        self.dataset_props['len_train_data'] = 600
        self.dataset_props['len_val_data'] = 100
        self.dataset_props['len_test_data'] = 100
        self.dataset_props['task'] = 'object_detection'
        self.dataset_props['layer_widths'] = layer_widths
        self.dataset_props['num_boxes'] = num_boxes
        self.test_batchsize = self.dataset_params['test_batchsize']

        print('Successfully loaded MNIST-objects dataset with input shape: {}'.format(self.dataset_props['input_shape']))

    self.load_task_losses_metrics() # initialize losses and metrics according to dataset_props['task']
