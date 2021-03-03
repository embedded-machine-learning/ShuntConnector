# -*- coding: utf-8 -*-
"""
Step #10 of the shunt connector procedure.

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

__author__ = 'Bernhard Haas'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'Apache 2.0'
__version__ = '1.0.0'
__maintainer__ = 'Bernhard Haas'
__email__ = 'bernhardhaas55@gmail.com'
__status__ = 'Release'

def print_summary(self):
    """This method represents step #10 of the shunt connection procedure.
       It prints the summary of the whole procedure to the log file.
       Includes information about FLOPS, accuracy and latency of the
       original, shunt and final model.
    """
    logging.info('')
    logging.info('')
    logging.info('#######################################################################################################')
    logging.info('############################################## SUMMARY ################################################')
    logging.info('#######################################################################################################')
    logging.info('')
    
    # Printing Accuracy
    logging.info('Accuracy:')
    logging.info('')
    for model_key in self.accuracy_dict.keys():
        logging.info('{}:'.format(model_key))
        for metric_key in self.accuracy_dict[model_key].keys():
            logging.info('{}: {}'.format(metric_key, self.accuracy_dict[model_key][metric_key]))
        logging.info('')
    
    # Printing Flopgs
    logging.info('FLOPs:')
    logging.info('')
    for model_key in self.flops_dict.keys():
        logging.info('{}:'.format(model_key))
        for metric_key in self.flops_dict[model_key].keys():
            logging.info('{}: {}'.format(metric_key, self.flops_dict[model_key][metric_key]))
        logging.info('')
    if 'original' in self.flops_dict.keys() and 'final' in self.flops_dict.keys():
        speed_up = 100*(self.flops_dict['original']['total']-self.flops_dict['final']['total'])/self.flops_dict['original']['total']
        logging.info('Speed up: {}'.format(speed_up))
        logging.info('')
    
    # Printing Latency
    if 'original_min' in self.latency_dict.keys() and 'final_min' in self.latency_dict.keys():
        logging.info('Latency:')
        logging.info('')
        for model_key in self.latency_dict.keys():
            logging.info('{}: {}'.format(model_key, self.latency_dict[model_key]))
            logging.info('')
        logging.info('Speed up min: {}'.format(100*(self.latency_dict['original_min']-self.latency_dict['final_min'])/self.latency_dict['original_min']))
        logging.info('Speed up median: {}'.format(100*(self.latency_dict['original_median']-self.latency_dict['final_median'])/self.latency_dict['original_median']))