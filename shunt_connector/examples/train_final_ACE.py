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

# Libs
from tensorflow import keras

# Own modules
import shunt_connector
from shunt_connector.utils import custom_loss_metric
from shunt_connector.utils.create_distillation_trainings_model import create_semantic_distillation_model

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
distillation_strength = 0.3

config_path = Path(sys.path[0], "config", "ACE.cfg")
config = configparser.ConfigParser()
config.read(config_path)

connector = shunt_connector.ShuntConnector(config)

connector.create_dataset()
connector.create_original_model()
connector.test_original_model()
connector.create_shunt_model()
connector.create_final_model()

# learning rate strategy
if connector.train_final_params['learning_policy'] == 'two_cycles':
    callback_learning_rate = shunt_connector.utils.custom_callbacks.LearningRateSchedulerCallback(epochs_first_cycle=connector.train_final_params['epochs_first_cycle'],
                                                            learning_rate_second_cycle=connector.train_final_params['learning_rate_second_cycle'])
elif connector.train_final_params['learning_policy'] == 'plateau':
    callback_learning_rate = keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                factor=connector.train_final_params['factor'],
                                                patience=connector.train_final_params['patience'],
                                                verbose=1,
                                                mode='auto',
                                                min_lr=1e-8)
elif connector.train_final_params['learning_policy'] == 'poly':
    callback_learning_rate = shunt_connector.utils.custom_callbacks.PolyLearningRateCallback(connector.train_final_params['power'],
                                                        connector.train_final_params['max_epochs'],
                                                        verbose=1)

# freezing strategy
if connector.train_final_params['freezing'] == 'nothing':
    pass
elif connector.train_final_params['freezing'] == 'freeze_before_shunt':
    for i, layer in enumerate(connector.final_model.layers):
        if i < connector.shunt_params['locations'][0]:
            layer.trainable = False

loss_dict = {}
metric_dict = {}

with connector.activate_distribution_scope():
    model_final_dist = create_semantic_distillation_model(connector.original_model, connector.final_model)
    loss_distillation = {model_final_dist.layers[-1].name: custom_loss_metric.create_ACE_loss(distillation_strength)}
    metrics_distillation = {model_final_dist.layers[-1].name: custom_loss_metric.ACE_metric(connector.dataset_props['num_classes'])}

callback_checkpoint = shunt_connector.utils.custom_callbacks.SaveNestedModelCallback('val_ace_metric', str(Path(connector.folder_name_logging, "final_model_weights.h5")), 'Student')
callbacks = [callback_checkpoint, callback_learning_rate]

with connector.distribute_strategy.scope():
    model_final_dist.compile(loss=loss_dict,
                        optimizer=keras.optimizers.SGD(lr=connector.train_final_params['base_learning_rate'], momentum=0.9, decay=0.0, nesterov=False),
                        metrics=metric_dict)

history_final = model_final_dist.fit(connector.dataset_train.batch(connector.train_final_params['batch_size']),
                                epochs=connector.train_final_params['max_epochs'],
                                steps_per_epoch=connector.dataset_props['len_train_data']//connector.train_final_params['batch_size'],
                                validation_data=connector.dataset_val.batch(connector.train_final_params['batch_size']),
                                validation_steps=connector.dataset_props['len_val_data']//connector.train_final_params['batch_size'],
                                verbose=1, 
                                callbacks=callbacks)

connector.final_model.load_weights(str(Path(connector.folder_name_logging, "final_model_weights.h5")))

keras.models.save_model(connector.final_model, Path(connector.folder_name_logging, "final_model.h5"))
