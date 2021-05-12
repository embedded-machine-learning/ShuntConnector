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
import numpy as np
import numpy.matlib

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

def create_CIFAR_dataset(num_classes=10, dataset_type='train'):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_CIFAR(num_classes=num_classes)

    if dataset_type == 'train':
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

    elif dataset_type == 'val':
        datagen = keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=False, 
            featurewise_std_normalization=False, 
            rotation_range=0.0,
            width_shift_range=0.0, 
            height_shift_range=0.0, 
            vertical_flip=False,
            horizontal_flip=False)
        ds = tf.data.Dataset.from_generator(lambda: datagen.flow(x_val, y_val, batch_size=1), output_types=(tf.float32, tf.float32), output_shapes = ([1,32,32,3],[1,num_classes]))
        ds = ds.unbatch()

    elif dataset_type == 'test':
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

    else:
        raise ValueError('Encountered invalid value for dataset_type!')

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

def create_MNIST_Objects_dataset(img_size, layer_widths, num_boxes, is_training=True):
    '''
    CODE FROM https://github.com/bruceyang2012/Face-detection-with-mobilenet-ssd
    '''

    TRAIN_SIZE = 600
    TEST_SIZE = 100

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[:TRAIN_SIZE, :, :]
    y_train = y_train[:TRAIN_SIZE]
    x_test = x_test[:TEST_SIZE, :, :]
    y_test = y_test[:TEST_SIZE]

    MNIST_SIZE = x_train.shape[-1]
    NUM_CLASSES = 10

    # Calculate anchor boxes

    total_num_boxes = sum([a*a*b for a,b in zip(layer_widths, num_boxes)])
    assert len(num_boxes) == len(layer_widths) # numBoxes for each layer and each layer has a specific width

    # number of scales is equal to the number of different resolutions ie num of layer widths
    # for a given resolution, we have different aspect ratios
    # num(scales) = num(layerWidth) = num(numBoxes) and num(asp_ratios) = numBoxes[i]
    min_scale = .1 # Min and Max scale given as percentage
    max_scale = 1.5
    scales = [ min_scale + x/len(layer_widths) * (max_scale-min_scale) for x in range(len(layer_widths)) ]
    scales = scales[::-1] # reversing the order because the layerWidths go from high to low (lower to higher resoltuion)

    asp = [0.5,1.0,1.5]
    asp1 = [x**0.5 for x in asp]
    asp2 = [1/x for x in asp1]

    centres = np.zeros((total_num_boxes,2))
    hw = np.zeros((total_num_boxes,2))
    boxes = np.zeros((total_num_boxes,4))
    
    # calculating the default box centres and height, width
    idx = 0

    for gridSize, numBox, scale in zip(layer_widths, num_boxes, scales):
        step_size = img_size*1.0/gridSize
        for i in range(gridSize):
            for j in range(gridSize):
                pos = idx + (i*gridSize+j) * numBox
                # centre is the same for all aspect ratios(=numBox)
                centres[ pos : pos + numBox , :] = i*step_size + step_size/2, j*step_size + step_size/2
                # height and width vary according to the scale and aspect ratio
                # zip asepct ratios and then scale them by the scaling factor
                hw[ pos : pos + numBox , :] = np.multiply(gridSize*scale, np.squeeze(np.dstack([asp1,asp2]),axis=0))[:numBox,:]

        idx += gridSize*gridSize*numBox

    boxes[:,0] = centres[:,0] - hw[:,0]/2
    boxes[:,1] = centres[:,1] - hw[:,1]/2
    boxes[:,2] = centres[:,0] + hw[:,0]/2
    boxes[:,3] = centres[:,1] + hw[:,1]/2

    # Generate GTs for all anchor boxes
    # calculate IoU for a set of search boxes and default boxes
    def IoU(box1, box2):
        box1 = box1.astype(np.float64)
        box2 = box2.astype(np.float64)
        # find the left and right co-ordinates of the edges. Min should be less than Max for non zero overlap
        xmin = np.maximum(box1[:,0],box2[:,0])
        xmax = np.minimum(box1[:,2],box2[:,2])
        ymin = np.maximum(box1[:,1],box2[:,1])
        ymax = np.minimum(box1[:,3],box2[:,3])

        intersection = np.abs(np.maximum(xmax-xmin,0) * np.maximum(ymax-ymin,0))
        boxArea1 = np.abs((box1[:,2] - box1[:,0]) * (box1[:,3] - box1[:,1]))
        boxArea2 = np.abs((box2[:,2] - box2[:,0]) * (box2[:,3] - box2[:,1]))
        unionArea = boxArea1 + boxArea2 - intersection
        assert (unionArea > 0).all()
        iou = intersection / unionArea

        return iou

    def bestIoU(search_box):
        return np.argwhere(IoU(numpy.matlib.repmat(search_box,total_num_boxes,1), boxes) > 0.5)

    def convert(x,y):
        # create a 2D array of top left corners for the mnist image to be placed
        corner = np.random.randint(img_size - MNIST_SIZE, size=(x.shape[0],2))

        # create a blank canvas for the input with the required dimension
        input = np.zeros((x.shape[0], img_size, img_size, 3))

        # replacing a part by RGB version of MNIST
        for i in range(x.shape[0]):
            lx = int(corner[i,0])
            ly = int(corner[i,1])
            input[i,lx:lx + MNIST_SIZE, ly:ly+MNIST_SIZE,:] = np.repeat(np.expand_dims(np.array(x[i,:,:]),axis=-1),3,axis=-1)

        # for each default box, there are 5 values: class number and delta cx,cy,h,w
        output = np.zeros((y.shape[0], total_num_boxes, 1+4))
        output[:,:,0] = NUM_CLASSES # defaulting class labels for all boxes to background initially
        for i in range(x.shape[0]):
            bbox = np.zeros(4)
            bbox[:2] = corner[i]
            bbox[2:] = corner[i] + (MNIST_SIZE,MNIST_SIZE)
            # for all default boxes which have IoU > threshold, set the delta values and class number
            box_idx = bestIoU(bbox).astype(np.uint16)
            output[i,box_idx,0] = y[i]
            output[i,box_idx,1] = (bbox[0] + bbox[2])/2.0 - centres[box_idx,0]
            output[i,box_idx,2] = (bbox[1] + bbox[3])/2.0 - centres[box_idx,1]
            output[i,box_idx,3] = MNIST_SIZE - hw[box_idx,0]
            output[i,box_idx,4] = MNIST_SIZE - hw[box_idx,1]

        return input, output    

    x_test, y_test = convert(x_test,y_test)
    x_train, y_train = convert(x_train,y_train)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).with_options(options)
    dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).with_options(options)

    return dataset_train, dataset_test