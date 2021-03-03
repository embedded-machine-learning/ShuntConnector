# -*- coding: utf-8 -*-
"""
Calculates knowledge quotients for each residual block of a given model.

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

# Own modules
from shunt_connector.utils.modify_model import modify_model
from shunt_connector.utils.keras_utils import identify_residual_layer_indexes

__author__ = 'Bernhard Haas'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'Apache 2.0'
__version__ = '1.0.0'
__maintainer__ = 'Bernhard Haas'
__email__ = 'bernhardhaas55@gmail.com'
__status__ = 'Release'

def get_knowledge_quotients(model, datagen, val_acc_model, metric=keras.metrics.categorical_accuracy, val_steps=None):

    know_quots = []
    add_input_dic, _ = identify_residual_layer_indexes(model)

    for add_layer_index in add_input_dic.keys():

        start_layer_index = add_input_dic[add_layer_index] + 1

        try:
            model_reduced = modify_model(model, range(start_layer_index, add_layer_index+1))
        except Exception as e:  # modify_model can fail for more complicated models, f.e. low scale feature addition in MobileNetV3-DeeplabV3
            print('Encountered following error while skipping block at index: {}: {}'.format(add_layer_index, str(e)))
            print('Skipping this block...')
            continue

        model_reduced.compile(metrics=[metric])

        if isinstance(datagen, tuple):
            _, val_acc = model_reduced.evaluate(datagen[0], datagen[1], steps=val_steps, verbose=1)
        else:
            _, val_acc = model_reduced.evaluate(datagen, steps=val_steps, verbose=1)

        print('Test accuracy for block {}: {:.5f}'.format(add_input_dic[add_layer_index], val_acc))

        know_quots.append((add_input_dic[add_layer_index], add_layer_index, 1 - val_acc/val_acc_model))

    return know_quots

