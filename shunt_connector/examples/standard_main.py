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

# Own modules
from shunt_connector import ShuntConnector

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


config_name = "standard.cfg"

if len(sys.argv) > 1:
    config_name = sys.argv[1]

config_path = Path(sys.path[0], "config", config_name)
config = configparser.ConfigParser()
config.read(config_path)

connector = ShuntConnector(config)
connector.execute()
