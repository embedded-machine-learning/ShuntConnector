# -*- coding: utf-8 -*-
"""
Step #1 of the shunt connection procedure.
License: TBD
"""

# Built-in/Generic Imports
from pathlib import Path

# Own modules
from shunt_connector.utils import custom_generators


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
        self.dataset_props['len_train_data'] = 50000
        self.dataset_props['len_val_data'] = 10000
        self.dataset_props['task'] = 'classification'
        self.test_batchsize = self.dataset_params['test_batchsize']

        self.dataset_train = custom_generators.create_CIFAR_dataset(num_classes=10,
                                                                    is_training=True)
        self.dataset_val = custom_generators.create_CIFAR_dataset(num_classes=10,
                                                                  is_training=False)

        print('CIFAR10 was loaded successfully!')

    if self.dataset_params['name'] == 'CIFAR100':

        self.dataset_props['num_classes'] = 100
        self.dataset_props['input_shape'] = (32,32,3)
        self.dataset_props['len_train_data'] = 50000
        self.dataset_props['len_val_data'] = 10000
        self.dataset_props['task'] = 'classification'
        self.test_batchsize = self.dataset_params['test_batchsize']

        self.dataset_train = custom_generators.create_CIFAR_dataset(num_classes=100,
                                                                    is_training=True)
        self.dataset_val = custom_generators.create_CIFAR_dataset(num_classes=100,
                                                                  is_training=False)
        print('CIFAR100 was loaded successfully!')

    if self.dataset_params['name'] == 'cityscapes':

        self.dataset_props['num_classes'] = 19
        self.dataset_props['input_shape'] = (self.dataset_params['input_size'][0],
                                             self.dataset_params['input_size'][1],
                                             3)
        self.dataset_props['len_train_data'] = 2975
        self.dataset_props['len_val_data'] = 500
        self.dataset_props['task'] = 'segmentation'
        self.test_batchsize = self.dataset_params['test_batchsize']

        self.dataset_train = custom_generators.create_cityscape_dataset(Path(self.dataset_params['path']),
                                                                             self.dataset_props['input_shape'], is_training=True)
        self.dataset_val = custom_generators.create_cityscape_dataset(Path(self.dataset_params['path']),
                                                                           self.dataset_props['input_shape'], is_training=False)
        print('Successfully loaded cityscapes dataset with input shape: {}'.format(self.dataset_props['input_shape']))

    self.load_task_losses_metrics() # initialize losses and metrics according to dataset_props['task']
