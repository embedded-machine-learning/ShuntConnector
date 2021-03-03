# -*- coding: utf-8 -*-
"""
Step #3 of the shunt connection procedure.

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
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Own modules
from shunt_connector.utils.custom_callbacks import LearningRateSchedulerCallback, PolyLearningRateCallback

__author__ = 'Bernhard Haas'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'Apache 2.0'
__version__ = '1.0.0'
__maintainer__ = 'Bernhard Haas'
__email__ = 'bernhardhaas55@gmail.com'
__status__ = 'Release'

def train_original_model(self):
    """This method represents step #3 of the shunt connection procedure.
       It traines the original model on the created dataset as defined in the loaded configuration file.

    Raises:
        ValueError: Raises an exception when configuration file does not hold information for the training of original model
        ValueError: Raises an exception when certain dataset propertiers got not set
        ValueError: Raises an exception when the original model got not set
    """
    print('\nTrain original model')

    if not self.train_original_params:
        raise ValueError('No parameters found in config for training original model! Create the field [TRAINING_ORIGINAL_MODEL].')
    if not self.dataset_train:
        raise ValueError('dataset_train field not initialized! Either call create_dataset or set it manually.')
    if not self.dataset_val:
        raise ValueError('dataset_val field  not initialized! Either call create_dataset or set it manually.')
    if not self.dataset_props['len_train_data']:
        raise ValueError('len_train_data field in dataset_props not initialized! Either call create_dataset or set it manually.')
    if not self.dataset_props['len_val_data']:
        raise ValueError('len_val_data field in dataset_props not initialized! Either call create_dataset or set it manually.')
    if not self.original_model:
        raise ValueError('original_model field not initialized! Either call create_original_model or set it manually.')

    with self.activate_distribution_scope():
        self.original_model.compile(loss=self.task_losses,
                                    metrics=self.task_metrics,
                                    optimizer=keras.optimizers.SGD(lr=self.train_original_params['base_learning_rate'],
                                    momentum=0.9,
                                    decay=0.0))

    # make sure that metrics_names field is initialized
    self.original_model.fit(self.dataset_train.batch(1),
                            epochs=1,
                            steps_per_epoch=1,
                            verbose=0)
    print('Using {} as the checkpoint monitor value!'.format("val_"+self.original_model.metrics_names[-1]))

    callback_checkpoint = keras.callbacks.ModelCheckpoint(str(Path(self.folder_name_logging, "original_model_weights.h5")),
                                                          save_best_only=True,
                                                          monitor="val_"+self.original_model.metrics_names[-1],
                                                          mode='max',
                                                          save_weights_only=True)

    # learning strategy
    if self.train_original_params['learning_policy'] == 'two_cycles':
        callback_learning_rate = LearningRateSchedulerCallback(epochs_first_cycle=self.train_original_params['epochs_first_cycle'],
                                                               learning_rate_second_cycle=self.train_original_params['learning_rate_second_cycle'])
    elif self.train_original_params['learning_policy'] == 'plateau':
        callback_learning_rate = ReduceLROnPlateau(monitor='loss',
                                                   factor=self.train_original_params['factor'],
                                                   patience=self.train_original_params['patience'],
                                                   verbose=1,
                                                   mode='auto',
                                                   min_lr=1e-8)
    elif self.train_original_params['learning_policy'] == 'poly':
        callback_learning_rate = PolyLearningRateCallback(self.train_original_params['power'],
                                                          self.train_original_params['max_epochs'],
                                                          verbose=1)
    else:
        raise ValueError('Encountered invalid learning_policy!')

    callbacks = [callback_checkpoint, callback_learning_rate]

    history_original = self.original_model.fit(self.dataset_train.batch(self.train_original_params['batch_size']),
                                            epochs=self.train_original_params['max_epochs'],
                                            steps_per_epoch=self.dataset_props['len_train_data']//self.train_original_params['batch_size'],
                                            validation_data=self.dataset_val.batch(self.train_original_params['batch_size']),
                                            validation_steps=self.dataset_props['len_val_data']//self.train_original_params['batch_size'],
                                            verbose=1,
                                            callbacks=callbacks)

    self.original_model.load_weights(str(Path(self.folder_name_logging, "original_model_weights.h5")))  # load best weights from training

    keras.models.save_model(self.original_model, Path(self.folder_name_logging, "original_model.h5"))
    logging.info('')
    logging.info('Original model saved to {}'.format(self.folder_name_logging))


def test_original_model(self):
    """This method represents an intermediate step of the shunt connection procedure.
       It tests the original model on the created dataset.

    Raises:
        ValueError: Raises an exception when certain dataset propertiers got not set
        ValueError: Raises an exception when the original model got not set
    """
    print('\nTest original model')

    if not self.dataset_train:
        raise Exception('Dataset not yet initialized! Have you called create_dataset?')
    if not self.original_model:
        raise Exception('Original model not yet initialized! Have you called create_original_model?')

    self.original_model.compile(loss=self.task_losses, metrics=self.task_metrics)

    metrics = self.original_model.evaluate(self.dataset_val.batch(self.test_batchsize),
                                           steps=self.dataset_props['len_val_data']//self.test_batchsize,
                                           verbose=1)

    self.accuracy_dict['original'] = {}
    for i, metric in enumerate(metrics):
        print('{}: {:.5f}'.format(self.original_model.metrics_names[i], metric))
        self.accuracy_dict['original'][self.original_model.metrics_names[i]] = metric
