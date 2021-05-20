# -*- coding: utf-8 -*-
"""
Step #9 of the shunt connection procedure.

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
import logging, timeit
import numpy as np
import gc

# Libs
import tensorflow as tf

__author__ = 'Bernhard Haas'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'Apache 2.0'
__version__ = '1.0.0'
__maintainer__ = 'Bernhard Haas'
__email__ = 'bernhardhaas55@gmail.com'
__status__ = 'Release'

def test_latency(self):
    print("\nTest latency")

    number_of_samples = self.test_latency_params['number_of_samples']

    random_input_tensor = tf.random.uniform((self.test_latency_params['batchsize'], self.original_model.input_shape[1], self.original_model.input_shape[2], self.original_model.input_shape[3]))

    print("warm up...")
    self.original_model(random_input_tensor)
    self.final_model(random_input_tensor)

    print("final model...")
    timings_final = np.asarray(timeit.repeat(lambda: self.final_model(random_input_tensor), number=20, repeat=self.test_latency_params['iterations'])) / number_of_samples
    print(timings_final)
    print("original model...")
    timings_original = np.asarray(timeit.repeat(lambda: self.original_model(random_input_tensor), number=20, repeat=self.test_latency_params['iterations'])) / number_of_samples
    print(timings_original)


    time_original_min = np.min(timings_original)
    time_final_min = np.min(timings_final)
    time_original_med = np.median(timings_original)
    time_final_med = np.median(timings_final)

    self.latency_dict['original_min'] = time_original_min
    self.latency_dict['final_min'] = time_final_min
    self.latency_dict['original_median'] = time_original_med
    self.latency_dict['final_median'] = time_final_med
