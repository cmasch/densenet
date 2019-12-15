# -*- coding: utf-8 -*-
"""
DenseNet implemented in TensorFlow 2

This implementation is based on the original paper of Gao Huang, Zhuang Liu, Kilian Q. Weinberger and Laurens van der Maaten.
Besides I took some influences by random implementations, especially of Zhuang Liu's Lua implementation.

# References
- [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
- [DenseNet - Lua implementation](https://github.com/liuzhuang13/DenseNet)

@author: Christopher Masch
"""

import tensorflow as tf
from tensorflow.keras import layers

__version__ = '0.1.0'

class DenseNet:

    def __init__(self, input_shape=None, dense_blocks=3, dense_layers=-1, growth_rate=12, nb_classes=None,
                 dropout_rate=None, bottleneck=False, compression=1.0, weight_decay=1e-4, depth=40):
        """
        Arguments:
            input_shape  : shape of the input images. E.g. (28,28,1) for MNIST
            dense_blocks : amount of dense blocks that will be created (default: 3)
            dense_layers : number of layers in each dense block. You can also use a list for numbers of layers [2,4,3]
                           or define only 2 to add 2 layers at all dense blocks. -1 means that dense_layers will be calculated
                           by the given depth (default: -1)
            growth_rate  : number of filters to add per dense block (default: 12)
            nb_classes   : number of classes
            dropout_rate : defines the dropout rate that is accomplished after each conv layer (except the first one).
                           In the paper the authors recommend a dropout of 0.2 (default: None)
            bottleneck   : (True / False) if true it will be added in convolution block (default: False)
            compression  : reduce the number of feature-maps at transition layer. In the paper the authors recomment a compression
                           of 0.5 (default: 1.0 - will have no compression effect)
            weight_decay : weight decay of L2 regularization on weights (default: 1e-4)
            depth        : number or layers (default: 40)
        """

        # Checks
        if nb_classes == None:
            raise Exception('Please define number of classes (e.g. nb_classes=10). This is required for final softmax.')

        if compression <= 0.0 or compression > 1.0:
            raise Exception('Compression have to be a value between 0.0 and 1.0.')

        if type(dense_layers) is list:
            if len(dense_layers) != dense_blocks:
                raise AssertionError('Number of dense blocks have to be same length to specified layers')
                
        elif dense_layers == -1:
            dense_layers = int((depth - 4) / 3)
            if bottleneck:
                dense_layers = int(dense_layers / 2)
            dense_layers = [dense_layers for _ in range(dense_blocks)]
        else:
            dense_layers = [dense_layers for _ in range(dense_blocks)]

        self.dense_blocks = dense_blocks
        self.dense_layers = dense_layers
        self.input_shape = input_shape
        self.growth_rate = growth_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.bottleneck = bottleneck
        self.compression = compression
        self.nb_classes = nb_classes

    def build_model(self):
        """
        Build the model

        Returns:
            Model : Keras model instance
        """

        print('Creating DenseNet')
        print('#############################################')
        print('Dense blocks: %s' % self.dense_blocks)
        print('Layers per dense block: %s' % self.dense_layers)
        print('#############################################')

        img_input = layers.Input(shape=self.input_shape, name='img_input')
        nb_channels = self.growth_rate

        # Initial convolution layer
        x = layers.Convolution2D(2 * self.growth_rate, (3, 3), padding='same', strides=(1, 1),
                                 kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))(img_input)

        # Building dense blocks
        for block in range(self.dense_blocks - 1):
            # Add dense block
            x, nb_channels = self.dense_block(x, self.dense_layers[block], nb_channels, self.growth_rate,
                                              self.dropout_rate, self.bottleneck, self.weight_decay)

            # Add transition_block
            x = self.transition_layer(x, nb_channels, self.dropout_rate, self.compression, self.weight_decay)
            nb_channels = int(nb_channels * self.compression)

        # Add last dense block without transition but for that with global average pooling
        x, nb_channels = self.dense_block(x, self.dense_layers[-1], nb_channels,
                                          self.growth_rate, self.dropout_rate, self.weight_decay)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        prediction = layers.Dense(self.nb_classes, activation='softmax')(x)

        return tf.keras.Model(inputs=img_input, outputs=prediction, name='densenet')

    def dense_block(self, x, nb_layers, nb_channels, growth_rate, dropout_rate=None, bottleneck=False,
                    weight_decay=1e-4):
        """
        Creates a dense block and concatenates inputs
        """

        for i in range(nb_layers):
            cb = self.convolution_block(x, growth_rate, dropout_rate, bottleneck)
            nb_channels += growth_rate
            x = layers.concatenate([cb, x])
        return x, nb_channels

    def convolution_block(self, x, nb_channels, dropout_rate=None, bottleneck=False, weight_decay=1e-4):
        """
        Creates a convolution block consisting of BN-ReLU-Conv.
        Optional: bottleneck, dropout
        """

        # Bottleneck
        if bottleneck:
            bottleneckWidth = 4
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Convolution2D(nb_channels * bottleneckWidth, (1, 1),
                                     kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
            # Dropout
            if dropout_rate:
                x = layers.Dropout(dropout_rate)(x)

        # Standard (BN-ReLU-Conv)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Convolution2D(nb_channels, (3, 3), padding='same')(x)

        # Dropout
        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)

        return x

    def transition_layer(self, x, nb_channels, dropout_rate=None, compression=1.0, weight_decay=1e-4):
        """
        Creates a transition layer between dense blocks as transition, which do convolution and pooling.
        Works as downsampling.
        """

        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Convolution2D(int(nb_channels * compression), (1, 1), padding='same',
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)

        # Adding dropout
        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)

        x = layers.AveragePooling2D((2, 2), strides=(2, 2))(x)
        return x
