# -*- coding: utf-8 -*-
"""
Calculates knowledge quotients for each residual block of a given model.
License: TBD
"""
# Libs
import tensorflow.keras as keras

# Own modules
from shunt_connector.utils.modify_model import modify_model
from shunt_connector.utils.keras_utils import identify_residual_layer_indexes

def get_knowledge_quotients(model, datagen, val_acc_model, metric=keras.metrics.categorical_accuracy, val_steps=None):

    know_quots = []
    add_input_dic, _ = identify_residual_layer_indexes(model)

    for add_layer_index in add_input_dic.keys():

        start_layer_index = add_input_dic[add_layer_index] + 1

        try:
            model_reduced = modify_model(model, range(start_layer_index, add_layer_index+1))
        except Exception as e:  # modify_model can fail for more complicated models, f.e. low scale feature addition in MobileNetV3-DeeplabV3
            print('Encountered following error while skipping block at index: {}: {}'.format(add_layer_index, str(e)))
            print('Skipping this block...')
            continue

        model_reduced.compile(metrics=[metric])

        if isinstance(datagen, tuple):
            _, val_acc = model_reduced.evaluate(datagen[0], datagen[1], steps=val_steps, verbose=1)
        else:
            _, val_acc = model_reduced.evaluate(datagen, steps=val_steps, verbose=1)

        print('Test accuracy for block {}: {:.5f}'.format(add_input_dic[add_layer_index], val_acc))

        know_quots.append((add_input_dic[add_layer_index], add_layer_index, 1 - val_acc/val_acc_model))

    return know_quots

