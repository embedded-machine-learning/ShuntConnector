# -*- coding: utf-8 -*-
"""
Custom callbacks for model training.

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
import numpy as np
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend

__author__ = 'Bernhard Haas'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'Apache 2.0'
__version__ = '1.0.0'
__maintainer__ = 'Bernhard Haas'
__email__ = 'bernhardhaas55@gmail.com'
__status__ = 'Release'

class LearningRateSchedulerCallback(Callback):

    def __init__(self, epochs_first_cycle, learning_rate_second_cycle):
        super().__init__()
        self.epochs_first_cycle = epochs_first_cycle
        self.learning_rate_second_cycle = learning_rate_second_cycle

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self.epochs_first_cycle:
            print("\nActivated second cycle with learning rate = {}\n".format(self.learning_rate_second_cycle))
            backend.set_value(self.model.optimizer.lr, self.learning_rate_second_cycle)

class SaveNestedModelCallback(Callback):

    def __init__(self, observed_value, weights_path, nested_model_name, mode='max'):
        super().__init__()
        self.observed_value = observed_value
        self.weights_path = weights_path
        self.nested_model_name = nested_model_name
        self.mode = mode
        if self.mode == 'max':
            self.best_value = 0
        if self.mode == 'min':
            self.best_value = 1e10

    def on_epoch_end(self, epoch, logs=None):
        new_value = logs[self.observed_value]
        if self.mode == 'max':
            if new_value > self.best_value:
                self.best_value = new_value
                self.model.get_layer(name=self.nested_model_name).save_weights(self.weights_path)
        elif self.mode == 'min':
            if new_value < self.best_value:
                self.best_value = new_value
                self.model.get_layer(name=self.nested_model_name).save_weights(self.weights_path)

class PolyLearningRateCallback(Callback):

    def __init__(self, power, max_epochs, verbose=False):
        super().__init__()
        self.power = power
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.base_lr = -1

    def on_epoch_end(self, epoch, logs=None):
        if self.base_lr == -1:
            self.base_lr = backend.get_value(self.model.optimizer.lr)
        lr = self.base_lr * np.power(1 - epoch / self.max_epochs, self.power)
        backend.set_value(self.model.optimizer.lr, lr)
        if self.verbose:
            print("\nChanged to learning rate: {}\n".format(lr))
