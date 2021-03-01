# -*- coding: utf-8 -*-
"""
Executes the whole shunt connector procedure.
License: TBD
"""

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
