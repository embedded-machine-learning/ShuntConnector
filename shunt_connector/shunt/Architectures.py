# -*- coding: utf-8 -*-
"""
Shunt Architectures.

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
# Libs
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, Add, Concatenate, Activation, ZeroPadding2D
from tensorflow.keras import Model
from tensorflow.keras import regularizers, initializers
from tensorflow import keras
from keras_applications import correct_pad

# Own modules
from shunt_connector.models.mobile_net_v3 import _se_block, _depth, hard_sigmoid

__author__ = 'Bernhard Haas'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'Apache 2.0'
__version__ = '1.0.0'
__maintainer__ = 'Bernhard Haas'
__email__ = 'bernhardhaas55@gmail.com'
__status__ = 'Release'

def createArch1(input_shape, output_shape, num_stride_layers, use_se, dilation_rates):

    input_net = Input(shape=input_shape)
    x = input_net

    x = Conv2D(192, kernel_size=(1,1), strides=(1,1), padding='same', use_bias=False, activation=None, name="shunt_conv_1", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name="shunt_batch_norm_1")(x)
    x = ReLU(6., name="shunt_relu_1")(x)
    if num_stride_layers > 0:
        x = ZeroPadding2D(padding=correct_pad(keras.backend, x, (3,3)), name='shunt_depthwise_pad_1')(x)
        x = DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='valid', use_bias=False, activation=None, name="shunt_depth_conv_1", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    else:
        x = DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), dilation_rate=(dilation_rates[0],dilation_rates[0]), padding='same', use_bias=False, activation=None, name="shunt_depth_conv_1", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name="shunt_batch_norm_2")(x)
    x = ReLU(6., name="shunt_relu_2")(x)
    x = Conv2D(64, kernel_size=(1,1), strides=(1,1), padding='same', use_bias=False, activation=None, name="shunt_conv_2", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name="shunt_batch_norm_3")(x)
    x = Conv2D(192, kernel_size=(1,1), strides=(1,1), padding='same', use_bias=False, activation=None, name="shunt_conv_3", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name="shunt_batch_norm_4")(x)
    x = ReLU(6., name="shunt_relu_3")(x)
    if num_stride_layers > 1:
        x = ZeroPadding2D(padding=correct_pad(keras.backend, x, (3,3)), name='shunt_depthwise_pad_2')(x)
        x = DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='valid', use_bias=False, activation=None, name="shunt_depth_conv_2", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    else:
        x = DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), dilation_rate=(dilation_rates[1],dilation_rates[1]), padding='same', use_bias=False, activation=None, name="shunt_depth_conv_2", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name="shunt_batch_norm_5")(x)
    x = ReLU(6., name="shunt_relu_4")(x)
    x = Conv2D(output_shape[-1], kernel_size=(1,1), strides=(1,1), padding='same', use_bias=False, activation=None, name="shunt_conv_4", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name="shunt_batch_norm_6")(x)

    model = Model(inputs=input_net, outputs=x, name='shunt')

    return model

def createArch4(input_shape, output_shape, num_stride_layers, use_se, dilation_rates):

    input_net = Input(shape=input_shape)
    x = input_net

    x = Conv2D(128, kernel_size=(1,1), strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = ReLU(6.)(x)
    if num_stride_layers > 0:
        x = DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    else:
        x = DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), dilation_rate=(dilation_rates[0],dilation_rates[0]), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = ReLU(6.)(x)
    x = Conv2D(output_shape[-1], kernel_size=(1,1), strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)

    model = Model(inputs=input_net, outputs=x, name='shunt')

    return model

def createArch5(input_shape, output_shape, num_stride_layers, use_se, dilation_rates):

    input_net = Input(shape=input_shape)
    x = input_net

    x = Conv2D(192, kernel_size=(1,1), strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = ReLU(6.)(x)
    if num_stride_layers > 0:
        x = DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    else:
        x = DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), dilation_rate=(dilation_rates[0],dilation_rates[0]), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = ReLU(6.)(x)
    x = Conv2D(64, kernel_size=(1,1), strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = Conv2D(192, kernel_size=(1,1), strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = ReLU(6.)(x)
    if num_stride_layers > 1:
        x = DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    else:
        x = DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), dilation_rate=(dilation_rates[1],dilation_rates[1]), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = ReLU(6.)(x)
    x = Conv2D(64, kernel_size=(1,1), strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = Conv2D(192, kernel_size=(1,1), strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = ReLU(6.)(x)
    if num_stride_layers > 2:
        x = DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    else:
        x = DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), dilation_rate=(dilation_rates[2],dilation_rates[2]), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = ReLU(6.)(x)
    x = Conv2D(output_shape[-1], kernel_size=(1,1), strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)

    model = Model(inputs=input_net, outputs=x, name='shunt')

    return model

def createShunt(input_shape, output_shape, arch, use_se=False, dilation_rate_input=1, dilation_rate_output=1, expansion_factor=6):

    assert(arch in [1,4,5,6])
    #assert(np.log2(input_shape[1] / output_shape[1]) == int(np.log2(input_shape[1] / output_shape[1])))

    model_shunt = None
    num_stride_layers = np.round(np.log2(input_shape[1] / output_shape[1]))
    max_stride_list = {1:2, 4:1, 5:3, 6:1}  # list of maximum strides for each architecture

    if max_stride_list[arch] < num_stride_layers:
        raise Exception("Chosen shunt architecture does not support {} many stride layers. Only {} are supported.".format(num_stride_layers, max_stride_list[arch]))

    # get dilation rates for given architecture
    dilation_rates = get_dilation_rates(arch, dilation_rate_input, dilation_rate_output)

    if arch == 1:
        model_shunt = createArch1(input_shape, output_shape, num_stride_layers, use_se, dilation_rates)
    if arch == 4:
        model_shunt = createArch4(input_shape, output_shape, num_stride_layers, use_se, dilation_rates)
    if arch == 5:
        model_shunt = createArch5(input_shape, output_shape, num_stride_layers, use_se, dilation_rates)
    if arch == 6:
        model_shunt = createArch6(input_shape, output_shape, num_stride_layers, use_se, expansion_factor, dilation_rates)

    return model_shunt

def get_dilation_rates(arch, dilation_rate_input, dilation_rate_output):

    dri = dilation_rate_input
    dro = dilation_rate_output

    if arch == 1:
        return [dri, dro]
    elif arch == 4:
        return [dro]
    elif arch == 5:
        return [dri, dri, dro]
    else:
        raise Exception("Unknown shunt architecture: {}".format(arch))