#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate standard configuration file for further developement.

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

__author__ = 'Bernhard Haas'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'Apache 2.0'
__version__ = '1.0.0'
__maintainer__ = 'Bernhard Haas'
__email__ = 'bernhardhaas55@gmail.com'
__status__ = 'Release'

def generate_standard_config():
    """ This function generates a standard config file 'standard.cfg' in the current directory.
    """

    config = configparser.ConfigParser(allow_no_value=True)
    config.optionxform=str

    # GENERAL
    config['GENERAL'] = {'train_original_model': False,
                            'test_original_model': False,
                            'calc_knowledge_quotients': False,
                            'train_shunt_model': False,
                            'test_shunt_model': False,
                            'train_final_model': False,
                            'test_final_model': False,
                            'test_fine-tune_strategies': False,
                            'test_latency': False}

    # DATASET
    config['DATASET'] = {'name': 'CIFAR10',
                            '# names: CIFAR10, CIFAR100, cityscapes': None,
                            'path': '',
                            'input_size': '32,32',
                            'test_batchsize': 1}

    # MODEL
    config['MODEL'] = {'type': 'DeeplabV3_MobileNetV3Small',
                        '# types: MobileNetV2, MobileNetV3Small, DeeplabV3_MobileNetV3Small': None,
                        'depth_factor': 1.0,
                        'from_file': False,
                        'filepath': "",
                        'pretrained': False,
                        '# pretrained: boolean for .h5 or \'imagenet\' or \'cityscapes\'': None,
                        'weightspath': "",
                        '# weightspath: h5 file or folder with .npy': None,
                        'output_stride': 32}

    # SHUNT
    config['SHUNT'] = {'locations': '64,157',
                        'arch': 1,
                        'from_file': False,
                        'filepath': "",
                        'pretrained': False,
                        'weightspath': ""}

    # FINAL MODEL
    config['FINAL_MODEL'] = {'test_after_shunt_insertion': False,
                             'pretrained': False,
                             'weightspath': ""}

    # TRAINING
    config['TRAINING_ORIGINAL_MODEL'] = {'learning_policy': 'poly',
                                            '# learning_policy: poly, plateau, two_cycles': None,
                                            'batchsize': 24,
                                            'max_epochs': 700,
                                            'base_learning_rate': 5e-2,
                                            'epochs_param': 4,
                                            'learning_rate_param': 0.9}

    config['TRAINING_SHUNT_MODEL'] = {'learning_policy': 'plateau',
                                        '# learning_policy: poly, plateau, two_cycles': None,
                                        'batchsize': 24,
                                        'max_epochs': 120,
                                        'base_learning_rate': 1e-1,
                                        'epochs_param': 4,
                                        'learning_rate_param': 1e-1}

    config['TRAINING_FINAL_MODEL'] = {'learning_policy': 'poly',
                                        '# learning_policy: poly, plateau, two_cycles': None,
                                        'freezing': 'nothing',
                                        '# freezing: nothing, freeze_before_shunt': None,
                                        'batchsize': 24,
                                        'max_epochs': 300,
                                        'base_learning_rate': 1e-2,
                                        'epochs_param': 4,
                                        'learning_rate_param': 0.9}

    config['TEST_LATENCY'] = {'iterations': 3,
                                'number_of_samples': 30,
                                'batchsize': 1}

    with open('standard.cfg', 'w') as configfile:
        config.write(configfile)

if __name__ == '__main__':
    generate_standard_config()
