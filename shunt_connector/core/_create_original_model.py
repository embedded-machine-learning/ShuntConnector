# -*- coding: utf-8 -*-
"""
Step #2 of the shunt connection procedure.

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
import logging
from pathlib import Path
from distutils.util import strtobool

# Libs
import tensorflow.keras as keras

# Own modules
from shunt_connector.models import mobile_net_v2, mobile_net_v3, deeplab_v3
from shunt_connector.utils import calculate_flops

__author__ = 'Bernhard Haas'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'Apache 2.0'
__version__ = '1.0.0'
__maintainer__ = 'Bernhard Haas'
__email__ = 'bernhardhaas55@gmail.com'
__status__ = 'Release'

def create_original_model(self):
    """This method represents step #2 of the shunt connection procedure.
       It creates the original model as defined in the loaded configuration file

    Raises:
        Exception: Raises an exception when configuration file does not hold information for the original model
        Exception: Raises an exception when certain dataset propertiers got not set
    """
    print('\nCreate original model')

    if not self.model_params:
        raise ValueError('No parameters found in config for creating original model! Create the field [MODEL]')
    if not self.dataset_props['num_classes']:
        raise ValueError('num_classes field in dataset_props not initialized! Either call create_dataset or set it manually.')
    if not self.dataset_props['input_shape']:
        raise ValueError('input_shape field in dataset_props not initialized! Either call create_dataset or set it manually.')
    if not self.task_losses:
        raise ValueError('task_losses field not initialized! Either call create_dataest or set it manually.')
    if not self.task_metrics:
        raise ValueError('task_metrics field not initialized! Either call create_dataest or set it manually.')

    logging.info('')
    logging.info('####################################################################################################')
    logging.info('########################################### ORIGINAL MODEL #########################################')
    logging.info('####################################################################################################')
    logging.info('')

    # parse pretrained parameter
    is_pretrained_custom = False
    is_pretrained_tf = None
    try:
        is_pretrained_custom = bool(strtobool(self.model_params['pretrained']))
    except ValueError:
        is_pretrained_tf = self.model_params['pretrained']


    with self.activate_distribution_scope():

        if self.model_params['from_file']:
            self.original_model = keras.models.load_model(self.model_params['filepath'])
        elif self.model_params['type'] == 'MobileNetV2':
            self.original_model = mobile_net_v2.create_mobilenet_v2(num_classes=self.dataset_props['num_classes'],
                                                                   input_shape=self.dataset_props['input_shape'],
                                                                   is_pretrained_on_imagenet=is_pretrained_tf=='imagenet',
                                                                   depth_factor=self.model_params['depth_factor'],
                                                                   num_change_strides=self.model_params['number_change_stride_layes'])

        elif self.model_params['type'] in ['MobileNetV3Small', 'MobileNetV3Large']:
            is_small = True
            if self.model_params['type'][11:] == 'Large':
                is_small = False

            self.original_model = mobile_net_v3.create_mobilenet_v3(is_pretrained_on_imagenet=is_pretrained_tf=='imagenet',
                                                                   num_classes=self.dataset_props['num_classes'],
                                                                   is_small=is_small,
                                                                   depth_factor=self.model_params['depth_factor'],
                                                                   input_shape=self.dataset_props['input_shape'],
                                                                   num_change_strides=self.model_params['number_change_stride_layes'])

        elif self.model_params['type'] == 'DeeplabV3_MobileNetV3Small':
            self.original_model = deeplab_v3.Deeplabv3(input_shape=(self.dataset_props['input_shape']),
                                                       depth_factor=self.model_params['depth_factor'],
                                                       classes=self.dataset_props['num_classes'],
                                                       OS=self.model_params['output_stride'],
                                                       weights=is_pretrained_tf,
                                                       tf_weightspath=self.model_params['weightspath'],
                                                       backbone='MobileNetV3')

        else: raise ValueError('Encountered invalid model type!')

    if is_pretrained_custom:
        self.original_model.load_weights(self.model_params['weightspath'])
        print('Weights loaded successfully!')

    # compile model with task specifid loss & metric
    with self.distribute_strategy.scope():
        self.original_model.compile(loss=self.task_losses, metrics=self.task_metrics)

    self.original_model.summary(print_fn=self.logger.info, line_length=150)
    print('{} created successfully!'.format(self.model_params['type']))

    keras.models.save_model(self.original_model, Path(self.folder_name_logging, "original_model.h5"))
    logging.info('')
    logging.info('Original model saved to %s',self.folder_name_logging)

    # calculate flops
    flops_original = calculate_flops.calculate_flops_model(self.original_model)
    print(flops_original)
    self.flops_dict['original'] = flops_original
    logging.info('')
    logging.info('FLOPs of original model: {}'.format(flops_original))

def set_original_model(self, model):
    """This method replaces step #2 of the shunt connection procedure.
       It should be used for defining a custom original model. It does NOT have to be compiled.

    Args:
        model (keras.models.Model): the model to be used as original model
    """
    print('\nSet original model')

    self.original_model = model
    
    keras.models.save_model(self.original_model, Path(self.folder_name_logging, "original_model.h5"))
    logging.info('')
    logging.info('Original model saved to %s',self.folder_name_logging)

    # calculate flops
    flops_original = calculate_flops.calculate_flops_model(self.original_model)
    self.flops_dict['original'] = flops_original
    logging.info('')
    logging.info('FLOPs of original model: %d', flops_original)
