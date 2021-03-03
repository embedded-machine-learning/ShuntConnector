# -*- coding: utf-8 -*-
"""
Custom generators used for training and evaluation of models.

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
from pathlib import Path

# Libs
import tensorflow as tf
import tensorflow.keras as keras

# Own modules
from shunt_connector.utils.dataset_utils import cityscapes_preprocess_image_and_label
from shunt_connector.utils.dataset_utils import load_and_preprocess_CIFAR

__author__ = 'Bernhard Haas'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'Apache 2.0'
__version__ = '1.0.0'
__maintainer__ = 'Bernhard Haas'
__email__ = 'bernhardhaas55@gmail.com'
__status__ = 'Release'

def create_CIFAR_dataset(num_classes=10, is_training=True):
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_CIFAR(num_classes=num_classes)

    if is_training:
        datagen = keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=False, 
            featurewise_std_normalization=False, 
            rotation_range=0.0,
            width_shift_range=0.2, 
            height_shift_range=0.2, 
            vertical_flip=False,
            horizontal_flip=True)

        ds = tf.data.Dataset.from_generator(lambda: datagen.flow(x_train, y_train, batch_size=1), output_types=(tf.float32, tf.float32), output_shapes = ([1,32,32,3],[1,num_classes]))
        ds = ds.unbatch()
        ds.shuffle(1000)
        ds.repeat()

    else:
        datagen = keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=False, 
            featurewise_std_normalization=False, 
            rotation_range=0.0,
            width_shift_range=0.0, 
            height_shift_range=0.0, 
            vertical_flip=False,
            horizontal_flip=False)
        ds = tf.data.Dataset.from_generator(lambda: datagen.flow(x_test, y_test, batch_size=1), output_types=(tf.float32, tf.float32), output_shapes = ([1,32,32,3],[1,num_classes]))
        ds = ds.unbatch()

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    ds = ds.with_options(options)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def create_cityscape_dataset(file_path, input_shape, is_training=True):
    if not isinstance(file_path, Path):     # convert str to Path
        file_path = Path(file_path)

    if is_training:
        preamble = 'train'
    else:
        preamble = 'val'
    record_file_list = list(map(str, file_path.glob(preamble + "*")))
    
    def parse_function(example_proto):
        def _decode_image(content, channels):
            return tf.cond(
                tf.image.is_jpeg(content),
                lambda: tf.image.decode_jpeg(content, channels),
                lambda: tf.image.decode_png(content, channels))

        features = {
            'image/encoded':
                tf.io.FixedLenFeature((), tf.string, default_value=''),
            'image/filename':
                tf.io.FixedLenFeature((), tf.string, default_value=''),
            'image/format':
                tf.io.FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/height':
                tf.io.FixedLenFeature((), tf.int64, default_value=0),
            'image/width':
                tf.io.FixedLenFeature((), tf.int64, default_value=0),
            'image/segmentation/class/encoded':
                tf.io.FixedLenFeature((), tf.string, default_value=''),
            'image/segmentation/class/format':
                tf.io.FixedLenFeature((), tf.string, default_value='png'),
        }

        parsed_features = tf.io.parse_single_example(example_proto, features)

        image = _decode_image(parsed_features['image/encoded'], channels=3)

        label = _decode_image(parsed_features['image/segmentation/class/encoded'], channels=1)

        image_name = parsed_features['image/filename']
        if image_name is None:
            image_name = tf.constant('')

        if label is not None:
            if label.get_shape().ndims == 2:
                label = tf.expand_dims(label, 2)
            elif label.get_shape().ndims == 3 and label.shape.dims[2] == 1:
                pass
            else:
                raise ValueError('Input label shape must be [height, width], or '
                            '[height, width, 1].')

        if not is_training:
            min_resize_value = input_shape[0]
            max_resize_value = input_shape[1]
        else:
            min_resize_value = None
            max_resize_value = None
     
        crop_height = input_shape[0]
        crop_width = input_shape[1]

        label.set_shape([None, None, 1])
        image, label = cityscapes_preprocess_image_and_label(image,
                                                             label,
                                                             crop_height=crop_height,
                                                             crop_width=crop_width,
                                                             min_resize_value=min_resize_value,
                                                             max_resize_value=max_resize_value,
                                                             is_training=is_training)

        return image, label

    ds = tf.data.TFRecordDataset(record_file_list, num_parallel_reads=tf.data.experimental.AUTOTUNE) \
         .map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if is_training:
        ds = ds.shuffle(100)
        ds = ds.repeat()
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    ds = ds.with_options(options)

    return ds
