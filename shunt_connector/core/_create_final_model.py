# -*- coding: utf-8 -*-
"""
Step #7 of the shunt connection procedure.

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
import logging

# Libs
import tensorflow.keras as keras

# Own modules
from shunt_connector.utils.modify_model import modify_model
from shunt_connector.utils.calculate_flops import calculate_flops_model

__author__ = 'Bernhard Haas'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'Apache 2.0'
__version__ = '1.0.0'
__maintainer__ = 'Bernhard Haas'
__email__ = 'bernhardhaas55@gmail.com'
__status__ = 'Release'

def create_final_model(self):
    """This method represents step #7 of the shunt connection procedure.
       It creates the final model by inserting the shunt model at the
       correct location instead of the original blocks.

    Raises:
        ValueError: Raises an exception when certain dataset propertiers got not set
        ValueError: Raises an exception when the original model got not set
    """
    print('\nCreate final model')

    if not self.original_model:
        raise ValueError('original_model field not initialized! Either call create_original_model or set it manually.')
    if not self.shunt_model:
        raise ValueError('shunt_model field not initialized! Either call create_shunt_model or set it manually.')
    if not self.shunt_params['locations']:
        raise ValueError('shunt_locations field not initialized! Either call create_shunt_model or set it manually.')

    logging.info('')
    logging.info('####################################################################################################')
    logging.info('########################################### FINAL MODEL ############################################')
    logging.info('####################################################################################################')
    logging.info('')

    with self.activate_distribution_scope():
        self.final_model = modify_model(self.original_model,
                                        layer_indexes_to_delete=range(self.shunt_params['locations'][0], self.shunt_params['locations'][1]+1), # +1 needed because of the way range works
                                        shunt_to_insert=self.shunt_model,
                                        shunt_location=self.shunt_params['locations'][0],
                                        layer_name_prefix='final_')

    for layer in self.final_model.layers:    # reset trainable status of all layers
        layer.trainable = True

    self.final_model.summary(print_fn=self.logger.info, line_length=150)

    # test shunt inserted model
    with self.activate_distribution_scope():
        self.final_model.compile(loss=self.task_losses,
                                 metrics=self.task_metrics,
                                 optimizer=keras.optimizers.SGD(lr=1e-10))

    # Sometimes the final model produces strange results for the first evaluation.
    # Can be fixed like this.
    metrics = self.final_model.evaluate(self.dataset_val.batch(1), steps=1, verbose=0)

    if self.final_model_params['test_after_shunt_insertion']:
        print('Test shunt inserted model')
        metrics = self.final_model.evaluate(self.dataset_val.batch(self.test_batchsize),
                                            steps=self.len_val_data//self.test_batchsize,
                                            verbose=1)

        self.accuracy_dict['shunt_inserted'] = {}
        for i, metric in enumerate(metrics):
            print('{}: {:.5f}'.format(self.final_model.metrics_names[i], metric))
            self.accuracy_dict['shunt_inserted'][self.final_model.metrics_names[i]] = metric

    if self.final_model_params['pretrained']:
        self.final_model.load_weights(self.final_model_params['weightspath'])
        print('Weights for final model loaded successfully!')

    self.final_model.trainable = True

    keras.models.save_model(self.final_model, Path(self.folder_name_logging, "final_model.h5"))
    logging.info('')
    logging.info('Final model saved to {}'.format(self.folder_name_logging))

    # calculate flops
    flops_final = calculate_flops_model(self.final_model)
    self.flops_dict['final'] = flops_final
    logging.info('')
    logging.info('FLOPs of final model: {}'.format(flops_final))

def set_final_model(self, model):
    """This method replaces step #7 of the shunt connection procedure.
       It should be used for defining a custom final model. It does NOT have to be compiled.

    Args:
        model (keras.models.Model): the model to be used as final model
    """
    print("\nSet final model")

    self.final_model = model

    keras.models.save_model(self.final_model, Path(self.folder_name_logging, "final_model.h5"))
    logging.info('')
    logging.info('Final model saved to {}'.format(self.folder_name_logging))

    # calculate flops
    flops_final = calculate_flops_model(self.final_model)
    self.flops_dict['final'] = flops_final
    logging.info('')
    logging.info('FLOPs of final model: {}'.format(flops_final))
