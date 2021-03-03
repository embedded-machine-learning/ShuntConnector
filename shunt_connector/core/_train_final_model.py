# -*- coding: utf-8 -*-
"""
Step #8 of the shunt connection procedure.

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

# Own modules
from shunt_connector.utils.custom_callbacks import SaveNestedModelCallback, LearningRateSchedulerCallback, PolyLearningRateCallback
from shunt_connector.utils.custom_loss_metric import create_mean_squared_diff_loss, create_ACE_loss, ACE_metric, create_negative_sum_loss
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

def train_final_model(self):
    """This method represents step #8 of the shunt connection procedure.
       It traines the final model on the created dataset as defined in the loaded configuration file.

    Raises:
        ValueError: Raises an exception when configuration file does not hold information for the training of final model
        ValueError: Raises an exception when certain dataset propertiers got not set
        ValueError: Raises an exception when the original model got not set
    """
    print('\nTrain final model')

    if not self.train_final_params:
        raise ValueError('No parameters found in config for training final model! Create the field [TRAINING_FINAL_MODEL].')
    if not self.dataset_train:
        raise ValueError('dataset_train field not initialized! Either call create_dataset or set it manually.')
    if not self.dataset_val:
        raise ValueError('dataset_val field  not initialized! Either call create_dataset or set it manually.')
    if not self.dataset_props['len_train_data']:
        raise ValueError('len_train_data field in dataset_props not initialized! Either call create_dataset or set it manually.')
    if not self.dataset_props['len_val_data']:
        raise ValueError('len_val_data field in dataset_props not initialized! Either call create_dataset or set it manually.')
    if not self.final_model:
        raise ValueError('final_model field not initialized! Either call create_final_model or set it manually.')

    # make sure that metrics_names field is initialized
    self.final_model.fit(self.dataset_train.batch(self.train_final_params['batch_size']),
                         epochs=1,
                         steps_per_epoch=1,
                         verbose=0)
    print('Using {} as the checkpoint monitor value!'.format("val_"+self.final_model.metrics_names[-1]))

    callback_checkpoint = keras.callbacks.ModelCheckpoint(str(Path(self.folder_name_logging, "final_model_weights.h5")),
                                                          save_best_only=True,
                                                          monitor="val_"+self.final_model.metrics_names[-1],
                                                          mode='max',
                                                          save_weights_only=True)
    self.final_model.trainable = True

    # learning rate strategy
    if self.train_final_params['learning_policy'] == 'two_cycles':
        callback_learning_rate = LearningRateSchedulerCallback(epochs_first_cycle=self.train_final_params['epochs_first_cycle'],
                                                               learning_rate_second_cycle=self.train_final_params['learning_rate_second_cycle'])
    elif self.train_final_params['learning_policy'] == 'plateau':
        callback_learning_rate = ReduceLROnPlateau(monitor='loss',
                                                   factor=self.train_final_params['factor'],
                                                   patience=self.train_final_params['patience'],
                                                   verbose=1,
                                                   mode='auto',
                                                   min_lr=1e-8)
    elif self.train_final_params['learning_policy'] == 'poly':
        callback_learning_rate = PolyLearningRateCallback(self.train_final_params['power'],
                                                          self.train_final_params['max_epochs'],
                                                          verbose=1)

    # freezing strategy
    if self.train_final_params['freezing'] == 'nothing':
        pass
    elif self.train_final_params['freezing'] == 'freeze_before_shunt':
        for i, layer in enumerate(self.final_model.layers):
            if i < self.shunt_locations[0]:  # TODO: TEST THIS!!
                layer.trainable = False

    loss_dict = self.task_losses
    metric_dict = self.task_metrics

    # callbacks
    callbacks = [callback_checkpoint, callback_learning_rate]

    with self.distribute_strategy.scope():
        self.final_model.compile(loss=loss_dict,
                            optimizer=keras.optimizers.SGD(lr=self.train_final_params['base_learning_rate'],momentum=0.9, decay=0.0, nesterov=False),
                            metrics=metric_dict)

    history_final = self.final_model.fit(self.dataset_train.batch(self.train_final_params['batch_size']),
                                    epochs=self.train_final_params['max_epochs'],
                                    steps_per_epoch=self.dataset_props['len_train_data']//self.train_final_params['batch_size'],
                                    validation_data=self.dataset_val.batch(self.train_final_params['batch_size']),
                                    validation_steps=self.dataset_props['len_val_data']//self.train_final_params['batch_size'],
                                    verbose=1,
                                    callbacks=callbacks)
                                    
    self.final_model.load_weights(str(Path(self.folder_name_logging, "final_model_weights.h5")))

    keras.models.save_model(self.final_model, Path(self.folder_name_logging, "final_model.h5"))
    logging.info('')
    logging.info('Final model saved to {}'.format(self.folder_name_logging))

def test_final_model(self):
    """This method represents an intermediate step of the shunt connection procedure.
       It tests the final model on the created dataset.

    Raises:
        ValueError: Raises an exception when certain dataset propertiers got not set
        ValueError: Raises an exception when the final model got not set
    """
    print('\nTest final model')

    if not self.dataset_train:
        raise Exception('Dataset not yet initialized! Have you called create_dataset?')
    if not self.final_model:
        raise Exception('Final model not yet initialized! Have you called create_shunt_model?')

    self.final_model.compile(loss=self.task_losses, metrics=self.task_metrics)

    metrics = self.final_model.evaluate(self.dataset_val.batch(self.test_batchsize),
                                        steps=self.dataset_props['len_val_data']//self.test_batchsize,
                                        verbose=1)
    self.accuracy_dict['final'] = {}
    for i, metric in enumerate(metrics):
        print('{}: {:.5f}'.format(self.final_model.metrics_names[i], metric))
        self.accuracy_dict['final'][self.final_model.metrics_names[i]] = metric
