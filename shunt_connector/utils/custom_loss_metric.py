# -*- coding: utf-8 -*-
"""
Custom losses and metrics used for training and evaulating models.

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
import unittest

# Libs
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

__author__ = 'Bernhard Haas'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'Apache 2.0'
__version__ = '1.0.0'
__maintainer__ = 'Bernhard Haas'
__email__ = 'bernhardhaas55@gmail.com'
__status__ = 'Release'

def create_negative_sum_loss(factor=1):
    def negative_sum(y_true, y_pred):
        return - factor*K.sum(y_pred)
    return negative_sum

def mean_abs_diff(y_true, y_pred):
    return K.mean(K.abs(y_pred))

def create_mean_squared_diff_loss(factor=1):
    def mean_squared_diff(y_true, y_pred):
        return factor*K.mean(K.square(y_pred))

    return mean_squared_diff

def segmentation_loss(y_true, y_pred):

    num_classes = K.int_shape(y_pred)[-1]

    y_true = tf.where(tf.equal(y_true, 255), 19 * tf.ones_like(y_true), y_true)
    y_true = tf.one_hot(tf.cast(y_true, tf.uint8), num_classes+1, axis=3)
    y_true = tf.squeeze(y_true, axis=-1)

    expansion = tf.zeros(shape=tf.shape(y_true)[:3], dtype=tf.float32)
    expansion = tf.expand_dims(expansion, axis=-1)
    y_pred = tf.concat((y_pred, expansion), axis=-1)
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

    weights = np.ones((num_classes+1,), dtype=np.float32)
    weights[-1] = 0
    weights = tf.convert_to_tensor(weights)

    loss = y_true * K.log(y_pred) * weights
    loss = -K.sum(loss, -1)
    return loss

class weighted_mean_iou(keras.metrics.MeanIoU):

    def __init__(self, num_classes):
        super().__init__(num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):

        pred_labels = tf.cast(K.argmax(y_pred, axis=-1), tf.float32)

        true_labels = tf.where(tf.equal(y_true, 255), 19 * tf.ones_like(y_true), y_true)

        weights = tf.cast(tf.not_equal(true_labels, 19), tf.float32)
        true_labels = tf.where(tf.equal(true_labels, 19), tf.zeros_like(true_labels), true_labels)

        super().update_state(true_labels, pred_labels, sample_weight=weights)

class ACE_metric(weighted_mean_iou):

    def __init__(self, num_classes):
        super().__init__(num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_student, _ = tf.split(y_pred, num_or_size_splits=2, axis=-1)
        super().update_state(y_true, y_student)

def create_ACE_loss(kappa):

    def ACE_loss(y_true, y_pred):

        y_student, y_teacher = tf.split(y_pred, num_or_size_splits=2, axis=-1)

        num_classes = K.int_shape(y_teacher)[-1]  

        labels_teacher = tf.expand_dims(tf.argmax(y_teacher, axis=-1), axis=-1)
        labels_true = tf.cast(y_true, tf.int64)

        expansion = tf.zeros(shape=tf.shape(y_true)[:3], dtype=tf.float32)
        expansion = tf.expand_dims(expansion, axis=-1)
        y_teacher_exp = tf.concat((y_teacher, expansion), axis=-1)

        y_true = tf.where(tf.equal(y_true, 255), 19 * tf.ones_like(y_true), y_true)
        y_true = tf.one_hot(tf.cast(y_true, tf.uint8), num_classes+1, axis=3)
        y_true = tf.squeeze(y_true, axis=-1)

        ace_true = tf.where(labels_teacher==labels_true, (kappa*y_teacher_exp + (1-kappa)*y_true), y_true)

        y_student = tf.concat((y_student, expansion), axis=-1)
        y_student = K.clip(y_student, K.epsilon(), 1 - K.epsilon())

        weights = np.ones((num_classes+1,), dtype=np.float32)
        weights[-1] = 0
        weights = tf.convert_to_tensor(weights)

        loss = ace_true * K.log(y_student) * weights
        loss = -K.sum(loss, -1)
        
        return loss
    
    return ACE_loss

class TestLossMetric(unittest.TestCase):

    def test_mean_iou(self):
        mean_iou = weighted_mean_iou(19)
        y_pred = tf.one_hot(tf.reshape(tf.constant([[1,1], [1,1]]), [1,2,2]), 19)
        y_true = tf.reshape(tf.constant([[1,1], [1,2]]), [1,2,2])
        mean_iou.update_state(y_true, y_pred)
        self.assertEqual(mean_iou.result(), np.mean([3/(3+1),0]))

    def test_ACE_metric(self):
        ace_metric = ACE_metric(19)
        y_pred = tf.one_hot(tf.reshape(tf.constant([[1,1], [1,1]]), [1,2,2]), 19)
        y_true = tf.reshape(tf.constant([[1,1], [1,2]]), [1,2,2])
        ace_metric.update_state(y_true, tf.concat([y_pred, tf.one_hot(y_true, 19)], axis=-1))
        print(ace_metric.result())
        self.assertEqual(ace_metric.result(), np.mean([3/(3+1),0]))  

    def test_segmentation_loss(self):
        y_pred = tf.one_hot(tf.reshape(tf.constant([[[1,1], [1,1]],[[1,1], [1,1]]]), [2,2,2]), 19)
        y_true = tf.reshape(tf.constant([[[1,1], [1,1]],[[1,1], [1,1]]]), [2,2,2,1])
        
        loss = np.asarray(segmentation_loss(y_true, y_pred))

        y_true = tf.one_hot(tf.cast(y_true, tf.uint8), 20, axis=3)
        y_true = tf.squeeze(y_true, axis=-1)
        
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
        y_pred[y_pred == 0] = 10e-8
        gt = - np.sum(y_true[...,:-1] * np.log(y_pred))
        self.assertTrue(np.sum(loss) - gt < 1e-6)


        y_pred = tf.one_hot(tf.reshape(tf.constant([[[2,1], [1,1]],[[1,1], [1,1]]]), [2,2,2]), 19)
        y_true = tf.reshape(tf.constant([[[1,1], [1,1]],[[1,1], [1,1]]]), [2,2,2,1])
        loss = np.asarray(segmentation_loss(y_true, y_pred))
        
        y_true = tf.one_hot(tf.cast(y_true, tf.uint8), 20, axis=3)
        y_true = tf.squeeze(y_true, axis=-1)
        
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
        y_pred[y_pred == 0] = 10e-8
        gt = - np.sum(y_true[...,:-1] * np.log(y_pred))
        self.assertTrue(np.sum(loss) - gt < 1e-6)


        y_pred = tf.one_hot(tf.reshape(tf.constant([[[18,18], [1,1]],[[1,1], [1,1]]]), [2,2,2]), 19)
        y_true = tf.reshape(tf.constant([[[19,19], [1,1]],[[1,1], [1,3]]]), [2,2,2,1])
        loss = np.asarray(segmentation_loss(y_true, y_pred))
        
        y_true = tf.one_hot(tf.cast(y_true, tf.uint8), 20, axis=3)
        y_true = tf.squeeze(y_true, axis=-1)

        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
        y_pred[y_pred == 0] = 10e-8
        gt = - np.sum(y_true[...,:-1] * np.log(y_pred))
        self.assertTrue(np.sum(loss) - gt < 1e-6)

    def test_ACE_loss(self):
        
        # teacher is right
        y_teacher = tf.one_hot(tf.reshape(tf.constant([[[1,1], [1,1]],[[1,1], [1,1]]]), [2,2,2]), 19)
        y_teacher = y_teacher * 0.8
        y_student = tf.one_hot(tf.reshape(tf.constant([[[2,2], [2,2]],[[2,2], [2,2]]]), [2,2,2]), 19)
        y_true = tf.reshape(tf.constant([[[1,1], [1,1]],[[1,1], [1,1]]]), [2,2,2,1])
        
        ACE_loss = create_ACE_loss(0.5, 1.0)

        loss = np.asarray(ACE_loss(y_true, tf.concat([y_student, y_teacher], axis=-1)))

        y_true = tf.one_hot(tf.cast(y_true, tf.uint8), 20, axis=3)
        y_true = tf.squeeze(y_true, axis=-1)
        expansion = tf.zeros(shape=tf.shape(y_true)[:3], dtype=tf.float32)
        expansion = tf.expand_dims(expansion, axis=-1)
        y_teacher = tf.concat((y_teacher, expansion), axis=-1)
        y_true = (tf.constant(1./3.) * y_teacher + tf.constant(2./3.) * y_true)
        
        y_true = y_true.numpy()
        y_pred = y_student.numpy()
        y_pred[y_pred == 0] = 10e-8
        gt = - np.sum(y_true[...,:-1] * np.log(y_pred))
        self.assertTrue(np.sum(loss) - gt < 1e-6)

        # teacher is wrong
        y_teacher = tf.one_hot(tf.reshape(tf.constant([[[2,2], [2,2]],[[2,2], [2,2]]]), [2,2,2]), 19)
        y_teacher = y_teacher * 0.8
        y_student = tf.one_hot(tf.reshape(tf.constant([[[2,2], [2,2]],[[2,2], [2,2]]]), [2,2,2]), 19)
        y_true = tf.reshape(tf.constant([[[1,1], [1,1]],[[1,1], [1,1]]]), [2,2,2,1])
        
        loss = np.asarray(ACE_loss(y_true, tf.concat([y_student, y_teacher], axis=-1)))

        y_true = tf.one_hot(tf.cast(y_true, tf.uint8), 20, axis=3)
        y_true = tf.squeeze(y_true, axis=-1)
        
        y_true = y_true.numpy()
        y_pred = y_student.numpy()
        y_pred[y_pred == 0] = 10e-8
        gt = - np.sum(y_true[...,:-1] * np.log(y_pred))
        self.assertTrue(np.sum(loss) - gt < 1e-6)



if __name__ == '__main__':

    unittest.main()
