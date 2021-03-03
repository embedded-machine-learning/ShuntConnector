# -*- coding: utf-8 -*-
"""
Implements the shunt connector object. Initialization file for the whole class.

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
import sys
import time
import logging
import configparser

# Libs
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import get_custom_objects

# Own modules
from shunt_connector.utils import custom_loss_metric
from shunt_connector.models import mobile_net_v3
from shunt_connector.core import _execute, _parse_config, _create_dataset, _create_original_model, _train_original_model, \
                                 _calc_knowledge_quotient, \
                                 _create_shunt_model, _train_shunt_model, _create_final_model, _train_final_model, \
                                 _test_latency, _print_summary

__author__ = 'Bernhard Haas'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'Apache 2.0'
__version__ = '1.0.0'
__maintainer__ = 'Bernhard Haas'
__email__ = 'bernhardhaas55@gmail.com'
__status__ = 'Release'

class ShuntConnector():

    _parse_config = _parse_config.parse_config
    execute = _execute.execute
    create_dataset = _create_dataset.create_dataset
    create_original_model = _create_original_model.create_original_model
    set_original_model = _create_original_model.set_original_model
    train_original_model = _train_original_model.train_original_model
    test_original_model = _train_original_model.test_original_model
    calc_knowledge_quotients = _calc_knowledge_quotient.calc_knowledge_quotients
    create_shunt_model = _create_shunt_model.create_shunt_model
    set_shunt_model = _create_shunt_model.set_shunt_model
    train_shunt_model = _train_shunt_model.train_shunt_model
    test_shunt_model = _train_shunt_model.test_shunt_model
    create_final_model = _create_final_model.create_final_model
    set_final_model = _create_final_model.set_final_model
    train_final_model = _train_final_model.train_final_model
    test_final_model = _train_final_model.test_final_model
    test_latency = _test_latency.test_latency
    print_summary = _print_summary.print_summary

    def __init__(self, config: configparser.ConfigParser):
        get_custom_objects().update({'hard_swish': mobile_net_v3.hard_swish,
                                     'hard_sigmoid': mobile_net_v3.hard_sigmoid})

        self.config = config

        # initialize param fields
        self.general_params = {}
        self.dataset_params = {}
        self.model_params = {}
        self.train_original_params = {}
        self.shunt_params = {}
        self.train_shunt_params = {}
        self.final_model_params = {}
        self.train_final_params = {}

        self._parse_config()

        # init logger
        self.folder_name_logging = Path(sys.path[0],
                                        "log",
                                        time.strftime("%Y%m%d"),
                                        time.strftime("%H_%M_%S"))
        Path(self.folder_name_logging).mkdir(parents=True, exist_ok=True)
        log_file_name = Path(self.folder_name_logging, "output.log")
        logging.basicConfig(filename=log_file_name, level=20, format='%(message)s')
        self.logger = logging.getLogger(__name__)

        # save config in output folder
        if config:
            with open( Path(self.folder_name_logging, "config.cfg"), 'w') as configfile:
                config.write(configfile)

        self.distribute_strategy = tf.distribute.MirroredStrategy(
                                   cross_device_ops=tf.distribute.ReductionToOneDevice())

        # init necessary fields
        self.dataset_val = None
        self.dataset_train = None
        self.original_model = None
        self.shunt_model = None
        self.shunt_trainings_model = None
        self.final_model = None
        self.task_losses = None
        self.task_metrics = None
        # settings which must be set
        self.dataset_props = {'task': None,
                              'num_classes': None,
                              'input_shape': None,
                              'len_train_data': None,
                              'len_val_data': None}
        self.test_batchsize = 1
        # measurements
        self.accuracy_dict = {}
        self.flops_dict = {}
        self.latency_dict = {}

    def activate_distribution_scope(self):
        """[Return the scope of the distribution strategy used in the ShuntConnector object.
            Should be used like: with shunt_connector.activate_distribution_scope():]

        Returns:
            [tf.distribution_strategy.scope]: [Distribution strategy scopse used in the ShuntConnector object]
        """
        return self.distribute_strategy.scope()

    def load_task_losses_metrics(self):
        """[Initializes losses and metrics used for all model operations based on the self.task field.]
        """
        if self.dataset_props['task'] is None:  # except if it was not found in config file
            return
        if self.dataset_props['task'] == 'classification':
            self.task_losses = ['categorical_crossentropy']
            with self.distribute_strategy.scope():
                self.task_metrics = [keras.metrics.categorical_crossentropy,
                                     keras.metrics.categorical_accuracy]
        elif self.dataset_props['task'] == 'segmentation':
            self.task_losses = [custom_loss_metric.segmentation_loss]
            with self.distribute_strategy.scope():
                self.task_metrics = [custom_loss_metric.weighted_mean_iou(19)]
        else: raise Exception('Task must be either classification|segmentation')
