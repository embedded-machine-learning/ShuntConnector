# -*- coding: utf-8 -*-
"""
Calculates the FLOPs (actually MACCs) for a given model or layer.
FLOPs are only counted for convolutional layers.
License: TBD
"""
# Built-in/Generic Imports
import imp

# Libs
# This is a hack to have the code work for tensorflow 2.2 as well as above
try:
    imp.find_module('tensorflow.python.keras.engine.functional.Functional')
    from tensorflow.python.keras.engine.functional import Functional
except:
    class Functional():
        pass
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D
from tensorflow.keras.models import Model

def calculate_flops_model(model):

    flops_dic = {'conv2d':0, 'depthwise_conv2d':0, 'total':0}

    for layer in model.layers:
        #print(layer.__class__)
        if layer.__class__ is Model or isinstance(layer, Functional):
            for layer_ in layer.layers:
                flops_dic = calculate_flops_layer(layer_, flops_dic)

        else:
            flops_dic = calculate_flops_layer(layer, flops_dic)

    return flops_dic

def calculate_flops_layer(layer, flops_dic):

    try:
        config = layer.get_config()
    except:
        return flops_dic

    flops_layer = 0
    if layer.__class__ is Conv2D:

        filters_input = layer.input_shape[3]
        image_size = (layer.input_shape[1], layer.input_shape[2])
        filters_output = config['filters']
        kernel_size = config['kernel_size'][0]
        strides = config['strides'][0]

        flops_layer = filters_input * image_size[0] * image_size[1] * filters_output * kernel_size * kernel_size / strides / strides
        flops_dic['total'] += flops_layer
        flops_dic['conv2d'] += flops_layer

    if layer.__class__ is DepthwiseConv2D:

        filters_input = layer.input_shape[3]
        image_size = (layer.input_shape[1], layer.input_shape[2])
        filters_output = filters_input
        kernel_size = config['kernel_size'][0]
        strides = config['strides'][0]

        flops_layer = filters_input * image_size[0] * image_size[1] * kernel_size * kernel_size / strides / strides
        flops_dic['total'] += flops_layer
        flops_dic['depthwise_conv2d'] += flops_layer


    return flops_dic
