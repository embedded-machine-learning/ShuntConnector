#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standard main script for the shunt connection procedure.

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
import configparser
from pathlib import Path
import sys

# Libs
import numpy as np
import bottleneck
from PIL import Image
from scipy.special import softmax
import tensorflow as tf
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Own modules
from shunt_connector import ShuntConnector

__author__ = 'Bernhard Haas'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'Apache 2.0'
__version__ = '1.0.0'
__maintainer__ = 'Bernhard Haas'
__email__ = 'bernhardhaas55@gmail.com'
__status__ = 'Release'
#-------------------------------------------------------------------------------------------------------------


config_name = "standard.cfg"

if len(sys.argv) > 1:
    config_name = sys.argv[1]

config_path = Path(sys.path[0], "config", config_name)
config = configparser.ConfigParser()
config.read(config_path)

connector = ShuntConnector(config)
connector.create_dataset()
connector.create_original_model()
connector.test_original_model()

OBJperCLASS = 10 # get the top 10 results for each class
outputChannels = 10 + 1 + 4
BOXES = 3150

img_size = 224
layer_widths = [28,14,7,4,2,1]
num_boxes = [3,3,3,3,3,3]
total_num_boxes = BOXES
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

X, Y = convert(x_test, y_test)

# get the confidence scores (with class values) and delta for the boxes. For each class, the top 10 values are used
def infer(Y):
    # classes are actually the index into the default boxes
    classes = np.zeros((OBJperCLASS,outputChannels-4),dtype=np.uint16)
    conf = np.zeros((OBJperCLASS,outputChannels-4))
    delta = np.zeros((OBJperCLASS,outputChannels-4,4))
    class_predictions = softmax(Y[:,:outputChannels-4],axis=1)
    for i in range(outputChannels-4):
        classes[:,i] = bottleneck.argpartition(class_predictions[:,i],BOXES-1-10,axis=-1)[-OBJperCLASS:]
        conf[:,i] = class_predictions[classes[:,i],i]
        delta[:,i] = Y[classes[:,i],outputChannels-4:]
    return conf,classes, delta

# generate bounding boxes from the inferred outputs
def Bbox(confidence, box_idx, delta):
    #delta contains delta(cx,cy,h,w)
    bbox_centre = np.zeros((OBJperCLASS,outputChannels-4,2))
    bbox_hw = np.zeros((OBJperCLASS,outputChannels-4,2))
    for i in range(OBJperCLASS):
        bbox_centre[i,:,0] = centres[box_idx[i]][:,0]+delta[i,:,0]
        bbox_centre[i,:,1] = centres[box_idx[i]][:,1]+delta[i,:,1]
        bbox_hw[i,:,0] = hw[box_idx[i]][:,0] + delta[i,:,2]
        bbox_hw[i,:,1] = hw[box_idx[i]][:,1]+delta[i,:,3]
    return bbox_centre,bbox_hw

y_pred = connector.original_model.predict(X)

for _ in range(10):
    r = np.random.randint(1, X.shape[0])

    print('Random int: ' + str(r))

    # top 10 predictions for each class
    confidence, box_idx, delta = infer(y_pred[r])
    bbox_centre,bbox_hw = Bbox(confidence, box_idx, delta)

    im = np.array(Image.fromarray(X[r].astype(np.uint8)))
    fig,ax = plt.subplots(1)
    ax.imshow(im)

    for i in range(outputChannels-4):
        # skipping backgrounds
        if i == NUM_CLASSES:
            continue
        color = 'r'
        # if a class is mentioned in the ground truth, color the boxes green
        if i in Y[r,:,0]:
            color = 'g'
            print(i)
        
        # skip all the classes which have low confidence values
        if (confidence[:,i] > 0.5).any() or i in Y[r,:,0]:
            for k in range(OBJperCLASS):
                print("{}: Confidence-{}\t\tCentre-{} Height,Width-{}".format(i,confidence[k,i],bbox_centre[k,i],bbox_hw[k,i]))
                
                # draw bounding box only if confidence scores are high
                if confidence[k,i] < 0.5:
                    continue
                x = bbox_centre[k,i,0] - bbox_hw[k,i,0]/2
                y = bbox_centre[k,i,1] - bbox_hw[k,i,1]/2
                rect = patches.Rectangle((y,x),bbox_hw[k,i,1],bbox_hw[k,i,0],linewidth=1,edgecolor=color,facecolor='none')
                ax.add_patch(rect)

    plt.show()