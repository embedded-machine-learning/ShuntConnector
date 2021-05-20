
# -*- coding: utf-8 -*-
"""
Step #6 of the shunt connection procedure.

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
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np

# Own modules
from shunt_connector.utils.custom_callbacks import SaveNestedModelCallback, LearningRateSchedulerCallback
from shunt_connector.utils.custom_loss_metric import mean_abs_diff, create_mean_squared_diff_loss
from shunt_connector.shunt.create_shunt_trainings_model import create_shunt_trainings_model

__author__ = 'Bernhard Haas'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'Apache 2.0'
__version__ = '1.0.0'
__maintainer__ = 'Bernhard Haas'
__email__ = 'bernhardhaas55@gmail.com'
__status__ = 'Release'

def train_shunt_model(self):
    """This method represents step #6 of the shunt connection procedure.
       It traines the shunt model on extracted feature maps as defined in the loaded configuration file

    Raises:
        ValueError: Raises an exception when configuration file does not hold information for the training of shunt model
        ValueError: Raises an exception when certain dataset propertiers got not set
        ValueError: Raises an exception when the original model got not set
    """
    print('\nTrain shunt model')

    if not self.dataset_train:
        raise ValueError('Dataset not yet initialized! Either call create_dataset or set it manually.')
    if not self.original_model:
        raise ValueError('Original model not yet initialized! Either call create_original_model or set it manually.')
    if not self.shunt_model:
        raise ValueError('Shunt model not yet initialized! Either call create_shunt_model or set it manually.')
    if not self.train_shunt_params:
        raise ValueError('No parameters found in config for training shunt model! Create the field [TRAINING_SHUNT_MODEL]')

    # create shunt model for training
    with self.activate_distribution_scope():
        loss_dict = { 'f_a_t': create_mean_squared_diff_loss(1) }
        metric_dict = {}

        self.shunt_training_model = create_shunt_trainings_model(self.original_model,
                                                                 self.shunt_model,
                                                                 self.shunt_params['locations'],
                                                                 mode='full_attention_transfer')
        self.shunt_training_model.compile(loss=loss_dict,
                                          optimizer=keras.optimizers.Adam(learning_rate=self.train_shunt_params['base_learning_rate'], decay=0.0),
                                          metrics=metric_dict)
    
    monitor_value_name = 'loss'
    if monitor_value_name.endswith('loss'):
        monitor_mode = 'min'
    else:
        monitor_mode = 'max'

    print('Using {} as the checkpoint monitor value with mode \'{}\'!'.format("val_"+monitor_value_name, monitor_mode))
    callback_checkpoint = SaveNestedModelCallback(weights_path=str(Path(self.folder_name_logging, "shunt_model_weights.h5")),
                                                  observed_value="val_"+monitor_value_name,
                                                  nested_model_name='shunt',
                                                  mode=monitor_mode)

    callback_learning_rate = ReduceLROnPlateau(monitor='loss',
                                               factor=0.1,
                                               patience=4,
                                               verbose=1,
                                               min_lr=1e-7)

    history_shunt = self.shunt_training_model.fit(self.dataset_train.batch(self.train_shunt_params['batch_size']),
                                                  epochs=self.train_shunt_params['max_epochs'],
                                                  steps_per_epoch=self.dataset_props['len_train_data']//self.train_shunt_params['batch_size'],
                                                  validation_data=self.dataset_val.batch(self.train_shunt_params['batch_size']),
                                                  validation_steps=self.dataset_props['len_val_data']//self.train_shunt_params['batch_size'],
                                                  verbose=1,
                                                  callbacks=[callback_checkpoint, callback_learning_rate])

    history_folder = Path(self.folder_name_logging) / Path('training_shunt_model_history')
    history_folder.mkdir(parents=True, exist_ok=True)

    for key in history_shunt.history.keys():
        np.save(history_folder / Path(key), history_shunt.history[key])

    self.shunt_model.load_weights(str(Path(self.folder_name_logging, "shunt_model_weights.h5"))) # load best weights from training

    keras.models.save_model(self.shunt_model, Path(self.folder_name_logging, "shunt_model.h5"))
    logging.info('')
    logging.info('Shunt model saved to {}'.format(self.folder_name_logging))

def test_shunt_model(self):
    """This method represents an intermediate step of the shunt connection procedure.
       It tests the shunt model fits the original model.

    Raises:
        ValueError: Raises an exception when certain dataset propertiers got not set
        ValueError: Raises an exception when the shunt model got not set
    """
    print('\nTest shunt model')

    if not self.shunt_trainings_model:
        with self.activate_distribution_scope():
            loss_dict = { 'f_a_t': create_mean_squared_diff_loss(1) }
            metric_dict = {}

            self.shunt_training_model = create_shunt_trainings_model(model=self.original_model,
                                                                     model_shunt=self.shunt_model,
                                                                     shunt_locations=self.shunt_params['locations'],
                                                                     mode='full_attention_transfer')

            self.shunt_training_model.compile(loss=loss_dict, metrics=metric_dict)

    if not self.dataset_train:
        raise Exception('Dataset not yet initialized! Have you called create_dataset?')
    if not self.shunt_model:
        raise Exception('Shunt model not yet initialized! Have you called create_shunt_model?')

    val_loss_shunt = self.shunt_training_model.evaluate(self.dataset_test.batch(self.test_batchsize),
                                                        steps=self.dataset_props['len_test_data']//self.test_batchsize,
                                                        verbose=1)

    print('Loss: {:.5f}'.format(val_loss_shunt))
    self.accuracy_dict['shunt'] = {'loss':val_loss_shunt}
