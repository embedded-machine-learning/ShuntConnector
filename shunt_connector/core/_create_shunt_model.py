# -*- coding: utf-8 -*-
"""
Step #5 of the shunt connection procedure.

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

# Libs
import tensorflow.keras as keras

# Own modules
from shunt_connector.shunt import Architectures
from shunt_connector.utils.calculate_flops import calculate_flops_model
from shunt_connector.shunt.find_dilation_rates import find_input_output_dilation_rates

__author__ = 'Bernhard Haas'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'Apache 2.0'
__version__ = '1.0.0'
__maintainer__ = 'Bernhard Haas'
__email__ = 'bernhardhaas55@gmail.com'
__status__ = 'Release'

def create_shunt_model(self):
    """This method represents step #5 of the shunt connection procedure.
       It creates the shunt model according to the options set in the config and properties of
       the original model like stride 2 layers and dilation rates.

    Raises:
        ValueError: Raises an exception when certain dataset propertiers got not set
        ValueError: Raises an exception when the original model got not set
    """

    print('\nCreate shunt model')

    if not self.original_model:
        raise ValueError('Original model not yet initialized! Either call create_original_model or set it manually.')
    if not self.shunt_params:
        raise ValueError('No parameters found in config for shunt model! Create the field [SHUNT_MODEL]')

    logging.info('')
    logging.info('#######################################################################################################')
    logging.info('############################################ SHUNT MODEL ##############################################')
    logging.info('#######################################################################################################')
    logging.info('')

    dilation_rate_input, dilation_rate_output = find_input_output_dilation_rates(self.original_model, self.shunt_params['locations'])

    print('Used dilation rates: {}'.format(Architectures.get_dilation_rates(self.shunt_params['arch'], dilation_rate_input, dilation_rate_output)))
    logging.info('Creating shunt with dilation rates: {}'.format(Architectures.get_dilation_rates(self.shunt_params['arch'], dilation_rate_input, dilation_rate_output)))
    logging.info('')

    with self.activate_distribution_scope():
        if self.shunt_params['from_file']:
            self.shunt_model = keras.models.load_model(self.shunt_params['filepath'])
            print('Shunt model loaded successfully!')
        else:
            input_shape_shunt = self.original_model.get_layer(index=self.shunt_params['locations'][0]).input_shape[1:]
            if isinstance(input_shape_shunt, list):
                input_shape_shunt = input_shape_shunt[0][1:]
            output_shape_shunt = self.original_model.get_layer(index=self.shunt_params['locations'][1]).output_shape[1:]
            if isinstance(output_shape_shunt, list):
                output_shape_shunt = output_shape_shunt[0][1:]
            self.shunt_model = Architectures.createShunt(input_shape_shunt,
                                                         output_shape_shunt,
                                                         arch=self.shunt_params['arch'],
                                                         use_se=False,
                                                         dilation_rate_input=dilation_rate_input,
                                                         dilation_rate_output=dilation_rate_output,
                                                         expansion_factor=1.0)

    if self.shunt_params['pretrained']:
        self.shunt_model.load_weights(self.shunt_params['weightspath'])
        print('Shunt weights loaded successfully!')

    self.shunt_model.summary(print_fn=self.logger.info, line_length=150)

    keras.models.save_model(self.shunt_model, Path(self.folder_name_logging, "shunt_model.h5"))
    logging.info('')
    logging.info('Shunt model saved to {}'.format(self.folder_name_logging))

    # calculate flops
    flops_shunt = calculate_flops_model(self.shunt_model)
    self.flops_dict['shunt'] = flops_shunt
    logging.info('')
    logging.info('FLOPs of shunt model: {}'.format(flops_shunt))


def set_shunt_model(self, model):
    """This method replaces step #4 of the shunt connection procedure.
       It should be used for defining a custom shunt model. It does NOT have to be compiled.

    Args:
        model (keras.models.Model): the model to be used as shunt model
    """
    print('\nSet shunt model')

    self.shunt_model = model

    keras.models.save_model(self.shunt_model, Path(self.folder_name_logging, "shunt_model.h5"))
    logging.info('')
    logging.info('Shunt model saved to {}'.format(self.folder_name_logging))

    # calculate flops
    flops_shunt = calculate_flops_model(self.shunt_model)
    self.flops_dict['shunt'] = flops_shunt
    logging.info('')
    logging.info('FLOPs of shunt model: {}'.format(flops_shunt))
