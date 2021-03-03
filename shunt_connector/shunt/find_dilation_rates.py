# -*- coding: utf-8 -*-
"""
Calculates the used dilation rates for a given model for the replaced blocks defined by
the shunt locations. Dilation rates are ONLY counted for depthwise-separable layers.

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

# Libs
import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D

__author__ = 'Bernhard Haas'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'Apache 2.0'
__version__ = '1.0.0'
__maintainer__ = 'Bernhard Haas'
__email__ = 'bernhardhaas55@gmail.com'
__status__ = 'Release'

def find_input_output_dilation_rates(model, shunt_locations):

    dilation_rates = []

    for i, layer in enumerate(model.layers):
        if shunt_locations[0] <= i <= shunt_locations[1]:
            if isinstance(layer, DepthwiseConv2D):
                config = layer.get_config()
                dilation_rates.append(config['dilation_rate'][0])

    dilation_rate_input = dilation_rates[0]
    dilation_rate_output = dilation_rates[-1]

    return dilation_rate_input, dilation_rate_output