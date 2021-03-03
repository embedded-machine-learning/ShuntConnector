# -*- coding: utf-8 -*-
"""
Implements the DeeplabV3-MobileNetV3Small model for semantic segmentation tasks.

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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Libs
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Lambda, Input, Activation,Add, BatchNormalization, AveragePooling2D, Multiply
from tensorflow.keras import backend as K

# Own modules
from shunt_connector.models.mobile_net_v3 import _inverted_res_block as _inverted_res_block_v3
from shunt_connector.models.mobile_net_v3 import hard_swish, hard_sigmoid, relu, _depth
from shunt_connector.utils.modify_model import modify_model

__author__ = 'Bernhard Haas'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'Apache 2.0'
__version__ = '1.0.0'
__maintainer__ = 'Bernhard Haas'
__email__ = 'bernhardhaas55@gmail.com'
__status__ = 'Release'

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def deeplab_head_v3(x, img_input, features_for_skip_connection):

    input_shape = K.int_shape(img_input)

    b4 = AveragePooling2D(pool_size=(13, 13), strides=(4,5), padding='valid')(x)
    b4 = Conv2D(128, (1, 1), padding='same', use_bias=True, name='image_pooling')(b4)
    b4 = Activation(hard_sigmoid, name='sigmoid_pooling')(b4)
    size_before = tf.keras.backend.int_shape(x)
    b4 = Lambda(lambda xx: tf.compat.v1.image.resize(xx,
                                                     size_before[1:3],
                                                     method='bilinear',
                                                     align_corners=True), name='Resize_1')(b4)

    b0 = Conv2D(128, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-3, momentum=0.999)(b0)
    b0 = Activation('relu', name='aspp0_relu')(b0)

    b = Multiply()([b0, b4])

    x = Lambda(lambda xx: tf.compat.v1.image.resize(xx,
                                                    (input_shape[1]//8+1,input_shape[2]//8+1),
                                                    method='bilinear',
                                                    align_corners=True), name='Resize_2')(b)
    x = Conv2D(19, (1, 1), padding='same', name='decoder_decoder_conv0')(x)

    additional_features = Conv2D(19,
                                 (1,1),
                                 padding='same',
                                 name='decoder_feature_projection0',
                                 use_bias=True)(features_for_skip_connection)

    x = Add()([x, additional_features])
    x = Lambda(lambda xx: tf.compat.v1.image.resize(xx,
                                                    (input_shape[1],input_shape[2]),
                                                    method='bilinear',
                                                    align_corners=True), name='Resize_3')(x)

    x = tf.keras.layers.Activation('softmax')(x)

    return x


def Deeplabv3(weights='cityscapes',
              tf_weightspath=None,
              input_shape=(1025, 2049, 3),
              classes=19,
              backbone='MobileNetV3',
              OS=16,
              depth_factor=1.0):

    """This method creates a DeeplabV3 model with the MobileNetV3 backbone, introduced in https://arxiv.org/abs/1905.02244.
       Weights can be loaded by specifying a folder containing weights in numpy array (.npy) form. This is done, so weights
       from a Tensorflow checkpoint file can be loaded. For this, the checkpoint file shall be converted to .npy files by 
       using the utils/convert_checkpoint_to_npy.py file.

    Args:
        weights (str): either 'cityscapes' or None. If 'cityscapes', the model loads the weights given as numpy arrays from the tf_weightspath.
        tf_weightspath (str): filepath of numpy arrays of weights for each layer 
        input_shape (tuple): tuple of the input shape of the model. Given in the format (H,W,C).
        classes (int): number of classes used by the semantic segmentation
        backbone (str): the backbone used as a feature extractor. Right now only 'MobileNetV3' is available.
        OS (int): output stride used by the feature extractor. Either [8,16,32].
        depth_factor (float): depth factor used for construction the MobileNet feature extractor. 

    Returns:
        keras.Model : the deeplabv3 model
    """

    assert OS in [8,16,32]

    img_input = Input(shape=input_shape)

    if backbone == 'MobileNetV3':

        def depth(d):
            return _depth(d * depth_factor)

        first_block_filters = _make_divisible(16, 8)
        x = Conv2D(first_block_filters,
                   kernel_size=3,
                   strides=(2, 2), padding='same',
                   use_bias=False, name='Conv')(img_input)
        x = BatchNormalization(
            epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
        x = Activation(hard_swish, name='Conv_HardSwish')(x)

        x = _inverted_res_block_v3(x, filters=depth(16), kernel_size=3, se_ratio=0.25, stride=2,
                                   expansion=1, block_id=0, activation=relu)

        x = _inverted_res_block_v3(x, filters=depth(24), kernel_size=3, se_ratio=None, stride=2,
                                   expansion=72./16, block_id=1, activation=relu)
        x, features_for_skip_connection = _inverted_res_block_v3(x, filters=depth(24), kernel_size=3, se_ratio=None, stride=1,
                                                                 expansion=88./24, block_id=2, activation=relu, return_expand_tensor=True)

        print('OS: ', OS)
        if OS == 32:
            rate = 1
            stride = 2
        elif OS == 16:
            rate = 1
            stride = 2
        elif OS == 8:
            rate = 2
            stride = 1

        x = _inverted_res_block_v3(x, filters=depth(40), kernel_size=5, se_ratio=0.25, stride=stride,
                                   expansion=4, block_id=3, activation=hard_swish)
        x = _inverted_res_block_v3(x, filters=depth(40), kernel_size=5, se_ratio=0.25, stride=1, rate=rate,
                                   expansion=6, block_id=4, activation=hard_swish)
        x = _inverted_res_block_v3(x, filters=depth(40), kernel_size=5, se_ratio=0.25, stride=1, rate=rate,
                                   expansion=6, block_id=5, activation=hard_swish)
        x = _inverted_res_block_v3(x, filters=depth(48), kernel_size=5, se_ratio=0.25, stride=1, rate=rate,
                                   expansion=3, block_id=6, activation=hard_swish)
        x = _inverted_res_block_v3(x, filters=depth(48), kernel_size=5, se_ratio=0.25, stride=1, rate=rate,
                                   expansion=3, block_id=7, activation=hard_swish)

        if OS == 32:
            rate_multiplier = 1
            stride = 2
        elif OS == 16 or OS == 8 :
            rate_multiplier = 2
            stride = 1

        x = _inverted_res_block_v3(x, filters=depth(48), kernel_size=5, se_ratio=0.25, stride=stride, rate=rate,
                                   expansion=3, block_id=8, activation=hard_swish, use_skip_connection=False)
        x = _inverted_res_block_v3(x, filters=depth(48), kernel_size=5, se_ratio=0.25, stride=1, rate=rate*rate_multiplier,
                                   expansion=6, block_id=9, activation=hard_swish)
        x = _inverted_res_block_v3(x, filters=depth(48), kernel_size=5, se_ratio=0.25, stride=1, rate=rate*rate_multiplier,
                                   expansion=6, block_id=10, activation=hard_swish)

        x = Conv2D(depth(288),
            kernel_size=1,
            strides=(1, 1), padding='same',
            use_bias=False, name='Conv_1')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_1_BN')(x)
        x = Activation(hard_swish, name='Conv_1_HardSwish')(x)

        x = deeplab_head_v3(x, img_input, features_for_skip_connection)

    model = Model(img_input, x, name='deeplabv3plus')

    if weights == 'cityscapes':
        load_cityscapes_weights(model, tf_weightspath)

    return modify_model(model, add_regularization=True)


def load_cityscapes_weights(model, weightspath):
    """This function loads weights in .npy form of a given filepath.

    Args:
        model (keras.Model): the Keras model for which weights should be loaded.
        weightspath (str): filepath to the folder.
    """

    for layer in model.layers:   
        if isinstance(layer, keras.models.Model):
            for nested_layer in layer.layers:
                if nested_layer.weights:
                    weights = []
                    for w in nested_layer.weights:
                        weight_name = os.path.basename(w.name).replace(':0', '')
                        weight_file = nested_layer.name + '_' + weight_name + '.npy'
                        weight_arr = np.load(os.path.join(weightspath, weight_file))
                        weights.append(weight_arr)
                    nested_layer.set_weights(weights)
        else:
            if layer.weights:
                weights = []
                for w in layer.weights:
                    weight_name = os.path.basename(w.name).replace(':0', '')
                    weight_file = layer.name + '_' + weight_name + '.npy'
                    weight_arr = np.load(os.path.join(weightspath, weight_file))
                    weights.append(weight_arr)
                layer.set_weights(weights)     
