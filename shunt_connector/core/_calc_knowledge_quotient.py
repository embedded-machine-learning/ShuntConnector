# -*- coding: utf-8 -*-
"""
Step #4 of the shunt connection procedure.

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

# Libs
from tensorflow import keras

# Own modules
from shunt_connector.utils.get_knowledge_quotients import get_knowledge_quotients

__author__ = 'Bernhard Haas'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'Apache 2.0'
__version__ = '1.0.0'
__maintainer__ = 'Bernhard Haas'
__email__ = 'bernhardhaas55@gmail.com'
__status__ = 'Release'

def calc_knowledge_quotients(self):
    """This method represents step #4 of the shunt connection procedure.
       It calculates the knowledge quotients of the original model by deleting each residual block inside the model
       and measure the relative change in accuracy.

    Raises:
        ValueError: Raises an exception when certain dataset propertiers got not set
        ValueError: Raises an exception when the original model got not set
    """

    print("\nCalc knowledge quotients")

    if not self.dataset_train:
        raise ValueError('Dataset not yet initialized! Have you called create_dataset?')
    if not self.dataset_props['len_val_data']:
        raise ValueError('len_val_data field in dataset_props not initialized! Either call create_dataset or set it manually.')
    if not self.original_model:
        raise ValueError('Original model not yet initialized! Either call create_original_model or set it manually.')

    if not 'original' in self.accuracy_dict:
        self.test_original_model()

    print(self.accuracy_dict)

    with self.activate_distribution_scope():
        self.know_quot = get_knowledge_quotients(model=self.original_model,
                                                 datagen=self.dataset_val.batch(self.test_batchsize),
                                                 val_acc_model=list(self.accuracy_dict['original'].items())[-1][1],
                                                 val_steps=self.dataset_props['len_val_data']//self.test_batchsize,
                                                 metric=self.task_metrics[-1])

    logging.info('')
    logging.info('################# KNOW QUOT RESULTS ###################')
    logging.info('')
    logging.info('')
    for (residual_idx, end_idx, value) in self.know_quot:
        logging.info("Block starts with: {}, location: {}".format(self.original_model.get_layer(index=residual_idx+1).name, residual_idx+1))
        logging.info("Block ends with: {}, location: {}".format(self.original_model.get_layer(index=end_idx).name, end_idx))   
        logging.info("Block knowledge quotient: {}\n".format(value)) 

