# -*- coding: utf-8 -*-
"""
Executes the whole shunt connector procedure.

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

__author__ = 'Bernhard Haas'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'Apache 2.0'
__version__ = '1.0.0'
__maintainer__ = 'Bernhard Haas'
__email__ = 'bernhardhaas55@gmail.com'
__status__ = 'Release'

def execute(self):
    """This method executes the whole shunt connector procedure according to the
    options set in the config file.

    Raises:
        ValueError: Raises an exception when configuration file does not hold 
        information about which steps should be executed
    """
    if not self.general_params:
        raise ValueError('GENERAL field not found in config!')

    self.create_dataset()
    self.create_original_model()
    if self.general_params['train_original_model']:
        self.train_original_model()
    if self.general_params['test_original_model']:
        self.test_original_model()
    if self.general_params['calc_knowledge_quotients']:
        self.calc_knowledge_quotients()
    self.create_shunt_model()
    if self.general_params['train_shunt_model']:
        self.train_shunt_model()
    if self.general_params['test_shunt_model']:
        self.test_shunt_model()
    self.create_final_model()
    if self.general_params['train_final_model']:
        self.train_final_model()
    if self.general_params['test_final_model']:
        self.test_final_model()
    if self.general_params['test_latency']:
        self.test_latency()
    self.print_summary()
