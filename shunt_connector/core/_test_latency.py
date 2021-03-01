# -*- coding: utf-8 -*-
"""
Step #9 of the shunt connection procedure.
License: TBD
"""
# Built-in/Generic Imports
import logging, timeit
import numpy as np
import gc

# Libs
import tensorflow as tf

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
