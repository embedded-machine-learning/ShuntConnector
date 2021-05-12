import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

from shunt_connector.models.mobile_net_v2 import create_mobilenet_v2

class SSD(keras.Model):
    def __init__(self, img_size, num_boxes=[4,6,6,6,4,4], layer_widths=[28,14,7,4,2,1], k = 10+1+4, num_change_strides=0):
        super(SSD,self).__init__()
        self.classes = k
        self.feature_maps = 6
        self.MobileNet = create_mobilenet_v2(input_shape=(img_size, img_size, 3), num_classes=k, num_change_strides=num_change_strides)

        self.num_boxes = num_boxes
        self.layer_widths = layer_widths
        self.features = [None for _ in range(self.feature_maps)]
        self.classifiers = [None for _ in range(self.feature_maps)]
        
        self.conv1_1 = layers.Conv2D(256,1,name='SSD_conv_1_1')
        self.conv1_2 = layers.Conv2D(512,3,strides=(2,2),padding='same',name='SSD_conv_1_2')

        self.conv2_1 = layers.Conv2D(128,1,name='SSD_conv_2_1')
        self.conv2_2 = layers.Conv2D(256,3,strides=(2,2),padding='same',name='SSD_conv_2_2')
        
        self.conv3_1 = layers.Conv2D(128,1,name='SSD_conv_3_1')
        self.conv3_2 = layers.Conv2D(256,3,strides=(1,1),name='SSD_conv_3_2')
        
        self.conv4_1 = layers.Conv2D(128,1,name='SSD_conv_4_1')
        self.conv4_2 = layers.Conv2D(256,2,strides=(1,1),name='SSD_conv_4_2') # changed the kernel size to 2 since the output of the previous layer has width 3

        self.conv = []
        self.reshape = []
        for i in range(self.feature_maps):
            self.conv.append(layers.Conv2D(self.num_boxes[i]*self.classes,3,padding='same',name='Classification_'+str(i)))
            self.reshape.append(layers.Reshape((self.layer_widths[i]* self.layer_widths[i] * self.num_boxes[i],self.classes),name='Reshape_classification_'+str(i)))

    def build(self, input_shape):
        self.MobileNet.build(input_shape)
    
    def call(self,inputs):
        x = inputs
        x = self.MobileNet(x)

        # get the convolved images at different resolutions
        self.features[0] = self.MobileNet.get_layer('expanded_conv_6_expand_relu').output
        self.features[1] = self.MobileNet.get_layer('expanded_conv_12_expand_relu').output
        self.features[2] = self.conv1_2(self.conv1_1(self.features[1]))
        self.features[3] = self.conv2_2(self.conv2_1(self.features[2]))
        self.features[4] = self.conv3_2(self.conv3_1(self.features[3]))
        self.features[5] = self.conv4_2(self.conv4_1(self.features[4]))

        for i in range(self.feature_maps):
            # for each feature map, create predictions according to the number of boxes for that layer and the number of output channels
            x = self.conv[i](self.features[i])
            x = self.reshape[i](x)
            self.classifiers[i] = x
            
        # concatenate all the classifiers
        x = layers.concatenate(self.classifiers, axis = -2, name='concatenate')
        return x


    def model(self):
        x = self.MobileNet.input
        return keras.Model(inputs=x, outputs=self.call(x))