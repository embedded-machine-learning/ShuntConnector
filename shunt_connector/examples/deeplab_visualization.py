#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization script for semantig segmentation tasks.

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
import sys
from pathlib import Path
import configparser

# Libs
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

# Own Modules
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


def create_cityscapes_label_colormap():
    """Creates a label colormap used in CITYSCAPES segmentation benchmark.
    Returns:
        A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    colormap[10] = [70, 130, 180]
    colormap[11] = [220, 20, 60]
    colormap[12] = [255, 0, 0]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]
    return colormap

if __name__ == '__main__':

    config_path = Path(sys.path[0], "config", "deeplab_vis.cfg")
    config = configparser.ConfigParser()
    config.read(config_path)

    connector = ShuntConnector(config)
    connector.create_dataset()
    connector.create_original_model()
    #connector.test_original_model()
    connector.create_shunt_model()
    connector.create_final_model()

    tf.random.set_seed(1235)
    data_val = connector.dataset_val.batch(1)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    data_val = data_val.with_options(options)

    color_map = create_cityscapes_label_colormap()

    i = 0

    for img, label in data_val.take(20):

        img = img + 1.0
        ax = plt.subplot(141)
        plt.imshow(img)
        ax.tick_params(labelbottom=False, labelleft=False)
        plt.title('Image')

        label = tf.where(tf.equal(label, 255), 19 * tf.ones_like(label), label)
        print(label.shape)
        ax = plt.subplot(142)
        plt.imshow(color_map[tf.squeeze(label)])
        ax.tick_params(labelbottom=False, labelleft=False)
        plt.title("Ground truth")

        pred = connector.original_model.predict(img)
        label = np.argmax(pred.squeeze(), axis=-1)

        ax = plt.subplot(143)
        plt.imshow(color_map[label])
        ax.tick_params(labelbottom=False, labelleft=False)
        plt.title('Original model')

        
        pred = connector.final_model.predict(img)
        label = np.argmax(pred.squeeze(), axis=-1)

        ax = plt.subplot(144)
        plt.imshow(color_map[label])
        ax.tick_params(labelbottom=False, labelleft=False)
        plt.title('Shunt inserted model')
        figure = plt.gcf()  # get current figure    
        figure.set_size_inches(16, 9) # set figure's size manually to your full screen (32x18)
        plt.savefig("cityscapes_1025_2049_original_model_{}".format(i))
        
        plt.show()
        i+=1

  