# -*- coding: utf-8 -*-
"""
Config parser of the shunt connector.
License: TBD
"""

def _check_param(argument: type, option: str, options: list):
    if not argument in options:
        raise ValueError("Provided argument {} for option {}, but only {} are legal arguments!".format(argument, option, options))
    return argument

def parse_config(self):

    config = self.config

    if 'GENERAL' in config.keys():
        self.general_params = {}
        self.general_params['task'] = config['GENERAL'].get('task')
        self.general_params['calc_knowledge_quotients'] = config['GENERAL'].getboolean('calc_knowledge_quotients')
        self.general_params['train_original_model'] = config['GENERAL'].getboolean('train_original_model')
        self.general_params['test_original_model'] = config['GENERAL'].getboolean('test_original_model')
        self.general_params['test_shunt_inserted_model'] = config['GENERAL'].getboolean('test_shunt_inserted_model')
        self.general_params['train_final_model'] = config['GENERAL'].getboolean('train_final_model')
        self.general_params['test_final_model'] = config['GENERAL'].getboolean('test_final_model')
        self.general_params['train_shunt_model'] = config['GENERAL'].getboolean('train_shunt_model')
        self.general_params['test_shunt_model'] = config['GENERAL'].getboolean('test_shunt_model')
        self.general_params['test_fine-tune_strategies'] = config['GENERAL'].getboolean('test_fine-tune_strategies')
        self.general_params['test_latency'] = config['GENERAL'].getboolean('test_latency')

    if 'DATASET' in config.keys():
        self.dataset_params = {}
        self.dataset_params['name'] = config['DATASET']['name']
        self.dataset_params['path'] = config['DATASET']['path']
        self.dataset_params['input_size'] = tuple(map(int, config['DATASET']['input_size'].split(',')))
        self.dataset_params['test_batchsize'] = config['DATASET'].getint('test_batchsize')

    if 'MODEL' in config.keys():
        self.model_params = {}
        self.model_params['type'] = config['MODEL']['type']
        self.model_params['depth_factor'] = config['MODEL'].getfloat('depth_factor')
        self.model_params['from_file'] = config['MODEL'].getboolean('from_file')
        self.model_params['filepath'] = config['MODEL']['filepath']
        self.model_params['pretrained'] = _check_param(config['MODEL'].get('pretrained'),
                                           'MODEL: pretrained', ['True', 'False', 'imagenet', 'cityscapes'])
        self.model_params['weightspath'] = config['MODEL']['weightspath']
        self.model_params['number_change_stride_layes'] = config['MODEL'].getint('change_stride_layers')
        self.model_params['output_stride'] = config['MODEL'].getint('output_stride')

    if 'TRAINING_ORIGINAL_MODEL' in config.keys():
        self.train_original_params = {}
        # obligatory parameters
        self.train_original_params['learning_policy'] = _check_param(config['TRAINING_ORIGINAL_MODEL'].get('learning_policy'),
                                                                              'TRAINING_ORIGINAL_MODEL: learning_policy', ['two_cycles', 'plateau', 'poly'])
        self.train_original_params['batch_size'] = config['TRAINING_ORIGINAL_MODEL'].getint('batchsize')
        self.train_original_params['max_epochs'] = config['TRAINING_ORIGINAL_MODEL'].getint('max_epochs')
        self.train_original_params['base_learning_rate'] = config['TRAINING_ORIGINAL_MODEL'].getfloat('base_learning_rate')
        # necessary parameters for given learning_strategy
        if self.train_original_params['learning_policy'] == 'two_cycles':
            self.train_original_params['epochs_first_cycle'] = config['TRAINING_ORIGINAL_MODEL'].getint('epochs_param')
            self.train_original_params['learning_rate_second_cycle'] = config['TRAINING_ORIGINAL_MODEL'].getfloat('learning_rate_param')
        if self.train_original_params['learning_policy'] == 'plateau':
            self.train_original_params['factor'] = config['TRAINING_ORIGINAL_MODEL'].getfloat('learning_rate_param')
            self.train_original_params['patience'] = config['TRAINING_ORIGINAL_MODEL'].getint('epochs_param')
        if self.train_original_params['learning_policy'] == 'poly':
            self.train_original_params['power'] = config['TRAINING_ORIGINAL_MODEL'].getfloat('learning_rate_param')

    if 'TRAINING_SHUNT_MODEL' in config.keys():
        self.train_shunt_params = {}
        self.train_shunt_params['learning_policy'] = _check_param(config['TRAINING_SHUNT_MODEL'].get('learning_policy'), 'TRAINING_SHUNT_MODEL: learning_policy', ['two_cycles', 'plateau', 'poly'])
        self.train_shunt_params['full_attention_transfer_factor'] = config['TRAINING_SHUNT_MODEL'].getfloat('full_attention_transfer_factor', 1.0)
        self.train_shunt_params['use_categorical_crossentropy'] = config['TRAINING_SHUNT_MODEL'].getboolean('use_categorical_crossentropy', False)
        self.train_shunt_params['batch_size'] = config['TRAINING_SHUNT_MODEL'].getint('batchsize')
        self.train_shunt_params['max_epochs'] = config['TRAINING_SHUNT_MODEL'].getint('max_epochs')
        self.train_shunt_params['base_learning_rate'] = config['TRAINING_SHUNT_MODEL'].getfloat('base_learning_rate')
        # necessary parameters for given learning_strategy
        if self.train_shunt_params['learning_policy'] == 'two_cycles':
            self.train_shunt_params['epochs_first_cycle'] = config['TRAINING_SHUNT_MODEL'].getint('epochs_param')
            self.train_shunt_params['learning_rate_second_cycle'] = config['TRAINING_SHUNT_MODEL'].getfloat('learning_rate_param')
        if self.train_shunt_params['learning_policy'] == 'plateau':
            self.train_shunt_params['factor'] = config['TRAINING_SHUNT_MODEL'].getfloat('learning_rate_param')
            self.train_shunt_params['patience'] = config['TRAINING_SHUNT_MODEL'].getint('epochs_param')
        if self.train_shunt_params['learning_policy'] == 'poly':
            self.train_shunt_params['power'] = config['TRAINING_SHUNT_MODEL'].getfloat('learning_rate_param')

    if 'TRAINING_FINAL_MODEL' in config.keys():
        self.train_final_params = {}
        # obligatory parameters
        self.train_final_params['learning_policy'] = _check_param(config['TRAINING_FINAL_MODEL'].get('learning_policy'), 'TRAINING_FINAL_MODEL: learning_policy', ['two_cycles', 'plateau', 'poly'])
        self.train_final_params['freezing'] = _check_param(config['TRAINING_FINAL_MODEL'].get('freezing'), 'TRAINING_FINAL_MODEL: freezing', ['nothing', 'freeze_before_shunt'])
        self.train_final_params['batch_size'] = config['TRAINING_FINAL_MODEL'].getint('batchsize')
        self.train_final_params['base_learning_rate'] = config['TRAINING_FINAL_MODEL'].getfloat('base_learning_rate')
        self.train_final_params['max_epochs'] = config['TRAINING_FINAL_MODEL'].getint('max_epochs')
        # necessary parameters for given learning_strategy
        if self.train_final_params['learning_policy'] == 'two_cycles':
            self.train_final_params['epochs_first_cycle'] = config['TRAINING_FINAL_MODEL'].getint('epochs_param')
            self.train_final_params['learning_rate_second_cycle'] = config['TRAINING_FINAL_MODEL'].getfloat('learning_rate_param')
        if self.train_final_params['learning_policy'] == 'plateau':
            self.train_final_params['factor'] = config['TRAINING_FINAL_MODEL'].getfloat('learning_rate_param')
            self.train_final_params['patience'] = config['TRAINING_FINAL_MODEL'].getint('epochs_param')
        if self.train_final_params['learning_policy'] == 'poly':
            self.train_final_params['power'] = config['TRAINING_FINAL_MODEL'].getfloat('learning_rate_param')

    if 'SHUNT' in config.keys():
        self.shunt_params = {}
        self.shunt_params['arch'] = config['SHUNT'].getint('arch')
        self.shunt_params['locations'] = tuple(map(int, config['SHUNT']['locations'].split(',')))
        self.shunt_params['from_file'] = config['SHUNT'].getboolean('from file')
        self.shunt_params['filepath'] = config['SHUNT']['filepath']
        self.shunt_params['pretrained'] = config['SHUNT'].getboolean('pretrained')
        self.shunt_params['weightspath'] = config['SHUNT']['weightspath']

    if 'FINAL_MODEL' in config.keys():
        self.final_model_params = {}
        self.final_model_params['test_after_shunt_insertion'] = config['FINAL_MODEL'].getboolean('test_after_shunt_insertion')
        self.final_model_params['pretrained'] = config['FINAL_MODEL'].getboolean('pretrained')
        self.final_model_params['weightspath'] = config['FINAL_MODEL']['weightspath']

    if 'TEST_LATENCY' in config.keys():
        self.test_latency_params = {}
        self.test_latency_params['iterations'] = config['TEST_LATENCY'].getint('iterations')
        self.test_latency_params['number_of_samples'] = config['TEST_LATENCY'].getint('number_of_samples')
        self.test_latency_params['batchsize'] = config['TEST_LATENCY'].getint('batchsize')
