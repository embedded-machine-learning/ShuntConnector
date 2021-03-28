# -*- coding: utf-8 -*-
"""
Dataset utils used by the custom generators.

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
import tensorflow as tf
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Own modules
import shunt_connector.utils.preprocess_utils_cityscapes as preprocess_utils

__author__ = 'Bernhard Haas'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'Apache 2.0'
__version__ = '1.0.0'
__maintainer__ = 'Bernhard Haas'
__email__ = 'bernhardhaas55@gmail.com'
__status__ = 'Release'

def load_and_preprocess_CIFAR(num_classes=10):
      
    if num_classes == 10:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif num_classes == 100:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    else:
        raise ValueError('Ecountered an invalid value for num_classes during loading of CIFAR dataset!')

    x_test = x_test.astype('float32')  #argmax_ch
    x_train = x_train.astype('float32')  #argmax_ch

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    def ch_wise_normalization(X_type, ch):
        mean_ch = x_train[:, :, :, ch].mean()
        std_ch = x_train[:, :, :, ch].std()
        X_type[:, :, :, ch] = (X_type[:, :, :, ch] - mean_ch) / std_ch
        return X_type[:, :, :, ch]

    x_test[:, :, :, 0]  = ch_wise_normalization(x_test, 0)
    x_test[:, :, :, 1]  = ch_wise_normalization(x_test, 1)
    x_test[:, :, :, 2]  = ch_wise_normalization(x_test, 2)
    x_val[:, :, :, 0]  = ch_wise_normalization(x_val, 0)
    x_val[:, :, :, 1]  = ch_wise_normalization(x_val, 1)
    x_val[:, :, :, 2]  = ch_wise_normalization(x_val, 2)
    x_train[:, :, :, 0]  = ch_wise_normalization(x_train, 0)
    x_train[:, :, :, 1]  = ch_wise_normalization(x_train, 1)
    x_train[:, :, :, 2]  = ch_wise_normalization(x_train, 2)

    y_val = to_categorical(y_val, num_classes)
    y_test  = to_categorical(y_test, num_classes)
    y_train = to_categorical(y_train, num_classes)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def cityscapes_preprocess_image_and_label(image,
                               label,
                               crop_height=769,
                               crop_width=769,
                               min_resize_value=None,
                               max_resize_value=None,
                               resize_factor=None,
                               min_scale_factor=0.5,
                               max_scale_factor=2.0,
                               scale_factor_step_size=0.25,
                               ignore_label=255,
                               is_training=True):
    """Preprocesses the image and label.
    Args:
        image: Input image.
        label: Ground truth annotation label.
        crop_height: The height value used to crop the image and label.
        crop_width: The width value used to crop the image and label.
        min_resize_value: Desired size of the smaller image side.
        max_resize_value: Maximum allowed size of the larger image side.
        resize_factor: Resized dimensions are multiple of factor plus one.
        min_scale_factor: Minimum scale factor value.
        max_scale_factor: Maximum scale factor value.
        scale_factor_step_size: The step size from min scale factor to max scale
        factor. The input is randomly scaled based on the value of
        (min_scale_factor, max_scale_factor, scale_factor_step_size).
        ignore_label: The label value which will be ignored for training and
        evaluation.
        is_training: If the preprocessing is used for training or not.
        model_variant: Model variant (string) for choosing how to mean-subtract the
        images. See feature_extractor.network_map for supported model variants.
    Returns:
        original_image: Original image (could be resized).
        processed_image: Preprocessed image.
        label: Preprocessed ground truth segmentation label.
    Raises:
        ValueError: Ground truth label not provided during training.
    """
    if is_training and label is None:
        raise ValueError('During training, label must be provided.')

    # Keep reference to original image.

    processed_image = tf.cast(image, tf.float32)

    if label is not None:
        label = tf.cast(label, tf.int32)

    # Resize image and label to the desired range.
    if min_resize_value or max_resize_value:
        [processed_image, label] = (
            preprocess_utils.resize_to_range(
                image=processed_image,
                label=label,
                min_size=min_resize_value,
                max_size=max_resize_value,
                factor=resize_factor,
                align_corners=True))
        
    # Data augmentation by randomly scaling the inputs.
    if is_training:
        scale = preprocess_utils.get_random_scale(
            min_scale_factor, max_scale_factor, scale_factor_step_size)
        processed_image, label = preprocess_utils.randomly_scale_image_and_label(
            processed_image, label, scale)
        processed_image.set_shape([None, None, 3])

    # Pad image and label to have dimensions >= [crop_height, crop_width]
    image_shape = tf.shape(processed_image)
    image_height = image_shape[0]
    image_width = image_shape[1]

    target_height = image_height + tf.maximum(crop_height - image_height, 0)
    target_width = image_width + tf.maximum(crop_width - image_width, 0)

    # Pad image with mean pixel value.
    mean_pixel = tf.reshape([127.5, 127.5, 127.5], [1, 1, 3])
    processed_image = preprocess_utils.pad_to_bounding_box(
        processed_image, 0, 0, target_height, target_width, mean_pixel)

    if label is not None:
        label = preprocess_utils.pad_to_bounding_box(
            label, 0, 0, target_height, target_width, ignore_label)

    # Randomly crop the image and label.
    if is_training and label is not None:
        processed_image, label = preprocess_utils.random_crop(
            [processed_image, label], crop_height, crop_width)

    processed_image.set_shape([crop_height, crop_width, 3])
    processed_image = tf.multiply(processed_image, 1/127.5)
    processed_image = tf.subtract(processed_image, 1)
    
    if label is not None:
        label.set_shape([crop_height, crop_width, 1])

    if is_training:
        # Randomly left-right flip the image and label.
        processed_image, label, _ = preprocess_utils.flip_dim(
            [processed_image, label], 0.5, dim=1)

    #label = tf.where(tf.equal(label, 255), 19 * tf.ones_like(label), label)
    #processed_image = tf.expand_dims(processed_image, 0)
    #label = tf.one_hot(label, 20, axis=2)
    #label = tf.squeeze(label, axis=-1)
    #label = tf.expand_dims(label, 0)

    return processed_image, label

