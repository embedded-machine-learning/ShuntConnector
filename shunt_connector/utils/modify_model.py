# -*- coding: utf-8 -*-
"""
Modifies a keras model according to given arguments.
License: TBD
"""
# Built-in/Generic Imports
from collections import OrderedDict

# Libs
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Activation, ReLU, Lambda, Softmax, ZeroPadding2D
from tensorflow.keras.layers import deserialize as layer_from_config

# Own modules
from shunt_connector.utils.keras_utils import get_index_of_layer

def modify_model(model,
                 layer_indexes_to_delete=[],
                 layer_indexes_to_output=[],
                 shunt_to_insert=None,
                 shunt_location=None,
                 layer_name_prefix="",
                 new_name_model=None,
                 change_stride_layers=0,
                 dense_softmax_to_seperate_softmax=False,
                 add_regularization=False,
                 transduce_weights=True):

    layer_outputs_dict = OrderedDict()

    outputs = []
    if 0 in layer_indexes_to_delete:
        if shunt_location in layer_indexes_to_delete:
            input_net = Input(shunt_to_insert.input_shape[1:], name=layer_name_prefix+model.layers[0].name)
        else:
            input_net = Input(model.layers[layer_indexes_to_delete[-1]].output_shape[1:],
                              name=layer_name_prefix+model.layers[0].name)
    else:
        input_net = Input(model.input_shape[1:], name=layer_name_prefix+model.layers[0].name)
    
    x = input_net
    layer_outputs_dict[layer_name_prefix+model.layers[0].name] = x

    for i in range(1,len(model.layers)):

        layer = model.layers[i]
        config = layer.get_config()
        config['name'] = layer_name_prefix + config['name']

        if add_regularization:
            if 'kernel_regularizer' in config.keys():
                config['kernel_regularizer'] = {'class_name': 'L1L2', 'config': {'l1': 0.0, 'l2': 4e-5}}
            if 'bias_regularizer' in config.keys():
                config['bias_regularizer'] = {'class_name': 'L1L2', 'config': {'l1': 0.0, 'l2': 4e-5}}

        if change_stride_layers > 0: 
            if isinstance(layer, ZeroPadding2D):
                config['padding'] = (1,1)
            if 'strides' in config.keys():
                if config['strides'] == (2,2): 
                    config['strides'] = (1,1)
                    change_stride_layers -= 1

        if dense_softmax_to_seperate_softmax:
            if i == len(model.layers)-1:            # last layer
                try:
                    config['activation'] = None
                except:
                    raise Exception('Tried to change activation of last layer to None, but layer config has no activation field!')

        next_layer = layer_from_config({'class_name': layer.__class__.__name__, 'config': config})


        if i in layer_indexes_to_delete:
            # insert shunt
            if i == shunt_location:
                x = shunt_to_insert(x)
                layer_outputs_dict['shunt'] = x

                if i in layer_indexes_to_output:
                    outputs.append(x)
            continue

        # insert shunt
        if i == shunt_location:
            x = shunt_to_insert(x)
            layer_outputs_dict['shunt'] = x
                
        input_layers = layer._inbound_nodes[-1].inbound_layers

        if not isinstance(input_layers, list):
            if get_index_of_layer(model, input_layers) in layer_indexes_to_delete:
                if shunt_to_insert:
                    x = next_layer(layer_outputs_dict['shunt'])
                else:
                    x = next_layer(x)
            else:
                x = next_layer(layer_outputs_dict[layer_name_prefix + input_layers.name])
        else:
            input_list = []
            for layer in input_layers:
                if get_index_of_layer(model, layer) in layer_indexes_to_delete:
                    if shunt_to_insert:
                        input_list.append(layer_outputs_dict['shunt'])
                    else:
                        input_list.append(list(layer_outputs_dict.items())[max(layer_indexes_to_delete[0]-1,0)][1])
                else:
                    input_list.append(layer_outputs_dict[layer_name_prefix + layer.name])

            x = next_layer(input_list)

        layer_outputs_dict[next_layer.name] = x

        if i in layer_indexes_to_output:
            #print(layer.name)
            outputs.append(x)

    if dense_softmax_to_seperate_softmax:
        x = Softmax(name='softmax')(x)

    outputs.append(x)
    assert(len(outputs) == len(layer_indexes_to_output)+1)

    if new_name_model:
        model_reduced = keras.models.Model(inputs=input_net, outputs=outputs, name=new_name_model)
    else:
        model_reduced = keras.models.Model(inputs=input_net, outputs=outputs, name=model.name)

    #print(model_reduced.summary())
    
    if not transduce_weights:
        return model_reduced

    for j in range(1,len(model_reduced.layers)):

        layer = model_reduced.layers[j]
        if isinstance(layer, ReLU) or isinstance(layer, Lambda) or isinstance(layer, Activation):
            continue

        # skip shunt
        if shunt_to_insert:
            if layer.name == 'shunt':
                layer.set_weights(shunt_to_insert.get_weights())
                continue

        if len(layer.get_weights()) > 0:
            weights = model.get_layer(name=layer.name[len(layer_name_prefix):]).get_weights()
            #print(layer.name)
            model_reduced.layers[j].set_weights(weights)

    return model_reduced
