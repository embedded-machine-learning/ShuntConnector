#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standard main script for the shunt connection procedure.

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
import configparser
from pathlib import Path
import sys
import logging

# Libs
from tensorflow import keras
import numpy as np

# Own modules
import shunt_connector

__author__ = 'Bernhard Haas'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'Apache 2.0'
__version__ = '1.0.0'
__maintainer__ = 'Bernhard Haas'
__email__ = 'bernhardhaas55@gmail.com'
__status__ = 'Release'
#-------------------------------------------------------------------------------------------------------------

# PARAMS
temperature = 5.0
distillation_strength = 4.0

config_path = Path("config", "dark_knowledge.cfg")
config = configparser.ConfigParser()
config.read(config_path)

connector = shunt_connector.ShuntConnector(config)

connector.create_dataset()
connector.create_original_model()
connector.test_original_model()
connector.create_shunt_model()
connector.test_shunt_model()
connector.create_final_model()


connector.logger.info('-------------------------------------------------------------------')
connector.logger.info('Distillation Params')
connector.logger.info('Temperature: {}, Strenght: {}'.format(temperature, distillation_strength))


from shunt_connector.utils import create_distillation_trainings_model
from shunt_connector.utils import custom_callbacks
from shunt_connector.utils import custom_loss_metric


# learning rate strategy
if connector.train_final_params['learning_policy'] == 'two_cycles':
    callback_learning_rate = custom_callbacks.LearningRateSchedulerCallback(epochs_first_cycle=connector.train_final_params['epochs_first_cycle'],
                                                            learning_rate_second_cycle=connector.train_final_params['learning_rate_second_cycle'])
elif connector.train_final_params['learning_policy'] == 'plateau':
    callback_learning_rate = keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                factor=connector.train_final_params['factor'],
                                                patience=connector.train_final_params['patience'],
                                                verbose=1,
                                                mode='auto',
                                                min_lr=1e-8)
elif connector.train_final_params['learning_policy'] == 'poly':
    callback_learning_rate = custom_callbacks.PolyLearningRateCallback(connector.train_final_params['power'],
                                                        connector.train_final_params['max_epochs'],
                                                        verbose=1)

# freezing strategy
if connector.train_final_params['freezing'] == 'nothing':
    pass
elif connector.train_final_params['freezing'] == 'freeze_before_shunt':
    for i, layer in enumerate(connector.final_model.layers):
        if i < connector.shunt_params['locations'][0]:  # TODO: TEST THIS!!
            layer.trainable = False

loss_dict = {}
metric_dict = {}

with connector.activate_distribution_scope():
    model_final_dist = create_distillation_trainings_model.create_classification_distillation_model(connector.final_model,
                                                                                                    connector.original_model,
                                                                                                    add_dark_knowledge=True,
                                                                                                    temperature=temperature)
    
    loss_dict = {'Student': 'categorical_crossentropy'}
    metric_dict = {'Student': ['accuracy']}
    callback_checkpoint = custom_callbacks.SaveNestedModelCallback('val_Student_accuracy', str(Path(connector.folder_name_logging, "final_model_weights.h5")), 'Student')
    for output in model_final_dist.output:
        output_name = output.name.split('/')[0] # cut off unimportant part
        if 'd_k' in output_name:
            loss_dict[output_name] = custom_loss_metric.create_weighted_cross_entropy_loss(distillation_strength)

callbacks = [callback_checkpoint, callback_learning_rate]

with connector.distribute_strategy.scope():
    model_final_dist.compile(loss=loss_dict,
                        optimizer=keras.optimizers.SGD(lr=connector.train_final_params['base_learning_rate'],momentum=0.9, decay=0.0, nesterov=False),
                        metrics=metric_dict)

history_final = model_final_dist.fit(connector.dataset_train.batch(connector.train_final_params['batch_size']),
                                epochs=connector.train_final_params['max_epochs'],
                                steps_per_epoch=connector.dataset_props['len_train_data']//connector.train_final_params['batch_size'],
                                validation_data=connector.dataset_val.batch(connector.train_final_params['batch_size']),
                                validation_steps=connector.dataset_props['len_val_data']//connector.train_final_params['batch_size'],
                                verbose=1, 
                                callbacks=callbacks)

history_folder = Path(connector.folder_name_logging) / Path('training_final_model_history')
history_folder.mkdir(parents=True, exist_ok=True)

for key in history_final.history.keys():
    np.save(history_folder / Path(key), history_final.history[key])

connector.final_model.load_weights(str(Path(connector.folder_name_logging, "final_model_weights.h5")))

keras.models.save_model(connector.final_model, Path(connector.folder_name_logging, "final_model.h5"))
logging.info('')
logging.info('Final model saved to {}'.format(connector.folder_name_logging))

connector.test_final_model()
connector.print_summary()