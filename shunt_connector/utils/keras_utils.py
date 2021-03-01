# -*- coding: utf-8 -*-
"""
Calculates the FLOPs (actually MACCs) for a given model or layer.
FLOPs are only counted for convolutional layers.
License: TBD
"""
# Libs
from tensorflow.keras.layers import Add, Multiply

def get_index_of_layer(model, layer):
    for i in range(len(model.layers)):
        if layer.name == model.layers[i].name:
            return i

def get_index_by_name(model, name):
    for i in range(len(model.layers)):
        if name == model.layers[i].name:
            return i

def get_first_layer_by_index(model, layers):
    smallest_index = len(model.layers)
    for layer in layers:
        index = get_index_of_layer(model, layer)
        if index < smallest_index:
            smallest_index = index
    return smallest_index

def identify_residual_layer_indexes(model):

    layers = model.layers
    add_incoming_index_dic = {}
    mult_incoming_index_dic = {}

    for i in range(len(layers)):

        layer = layers[i]

        if isinstance(layer, Add):
            input_layers = layer._inbound_nodes[-1].inbound_layers
            incoming_index = get_first_layer_by_index(model, input_layers)
            add_incoming_index_dic[i] = incoming_index

        if isinstance(layer, Multiply):
            input_layers = layer._inbound_nodes[-1].inbound_layers
            incoming_index = get_first_layer_by_index(model, input_layers)
            mult_incoming_index_dic[i] = incoming_index

    return add_incoming_index_dic, mult_incoming_index_dic
