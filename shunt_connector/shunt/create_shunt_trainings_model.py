# -*- coding: utf-8 -*-
"""
Implements the trainings model for a given shunt architecture.

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
import tensorflow.keras as keras
from tensorflow.keras.layers import Subtract, Flatten, Multiply

# Own modules
from shunt_connector.utils.modify_model import modify_model

__author__ = 'Bernhard Haas'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'Apache 2.0'
__version__ = '1.0.0'
__maintainer__ = 'Bernhard Haas'
__email__ = 'bernhardhaas55@gmail.com'
__status__ = 'Release'

def create_shunt_trainings_model(model, model_shunt, shunt_locations, mode):

    if mode == 'full_attention_transfer':
        return _full_attention_trainings_model(model, model_shunt, shunt_locations)
    elif mode == 'categorical_crossentropy':
        return _categorical_trainings_model(model, model_shunt, shunt_locations)
    else:
        raise Exception('Unknonw shunt trainings mode!')

def _categorical_trainings_model(model, model_shunt, shunt_locations):
    
    shunt_input = model.layers[shunt_locations[0]-1].output

    output_original_model = model.layers[shunt_locations[1]].output
    output_original_model_flattend = Flatten()(output_original_model)

    # create shunt model with prediction output
    prediction_head = modify_model(model, layer_indexes_to_delete=range(0,shunt_locations[1]+1), new_name_model='cross_entropy')

    shunt_output = model_shunt(shunt_input)
    prediction_shunt = prediction_head(shunt_output)
    shunt_output_flattend = Flatten()(shunt_output)
    shunt_loss = Subtract(name='f_a_t')([shunt_output_flattend, output_original_model_flattend])

    model_training = keras.models.Model(inputs=model.input, outputs=[shunt_loss, prediction_shunt], name='shunt_training')
    
    # set all layers untrainable, except shunt layers
    for layer in model_training.layers: layer.trainable = False
    model_training.get_layer(name='shunt').trainable = True

    return model_training

def _full_attention_trainings_model(model, model_shunt, shunt_locations):
   
    shunt_input = model.layers[shunt_locations[0]-1].output

    output_original_model = model.layers[shunt_locations[1]].output
    output_original_model_flattend = Flatten()(output_original_model)

    shunt_output = model_shunt(shunt_input)
    shunt_output_flattend = Flatten()(shunt_output)
    shunt_loss = Subtract(name='f_a_t')([shunt_output_flattend, output_original_model_flattend])

    model_training = keras.models.Model(inputs=model.input, outputs=[shunt_loss], name='shunt_training')
    
    # set all layers untrainable, except shunt layers
    for layer in model_training.layers: layer.trainable = False
    model_training.get_layer(name='shunt').trainable = True

    return model_training