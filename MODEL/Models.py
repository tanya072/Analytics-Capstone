import os
import sys
import glob
import argparse
from keras import __version__ #
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from keras.models import *
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, \
    Activation, ZeroPadding2D
from keras.layers import add, Flatten
from keras import backend as K, layers
from keras_applications.imagenet_utils import _obtain_input_shape
import warnings
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications import ResNet50
from keras.utils import plot_model
from keras import regularizers


class Models():
    def __init__(self, classes, train_data, train_labels, test_data, test_labels, class_list):
        self.classes = classes
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.class_list = class_list

    @staticmethod
    def seq_setting(norm_size):
        model = Sequential()

        model.add(Dense(512, input_shape=(norm_size,), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        #model.add(Dropout(0.5))
        #model.add(Dense(128, activation='relu'))
        #model.add(Dropout(0.5))
        #model.add(Dense(68, activation='relu'))
        model.add(Dense(5, activation='softmax'))

        init_lr = 0.001
        opt = SGD(lr=init_lr)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    def Conv2d_BN(self, x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
        x = BatchNormalization(axis=3, name=bn_name)(x)
        return x

    def identity_Block(self, inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
        x = self.Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
        x = self.Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
        if with_conv_shortcut:
            shortcut = self.Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
            x = add([x, shortcut])
            return x
        else:
            x = add([x, inpt])
            return x

    def bottleneck_Block(self, inpt, nb_filters, strides=(1, 1), with_conv_shortcut=False):
        k1, k2, k3 = nb_filters
        x = self.Conv2d_BN(inpt, nb_filter=k1, kernel_size=1, strides=strides, padding='same')
        x = self.Conv2d_BN(x, nb_filter=k2, kernel_size=3, padding='same')
        x = self.Conv2d_BN(x, nb_filter=k3, kernel_size=1, padding='same')
        if with_conv_shortcut:
            shortcut = self.Conv2d_BN(inpt, nb_filter=k3, strides=strides, kernel_size=1)
            x = add([x, shortcut])
            return x
        else:
            x = add([x, inpt])
            return x

    #resnet_34
    def Res_setting(self, width, height, channel, classes, opt=SGD(lr=0.001), dp=0):
        inpt = Input(shape=(width, height, channel))
        x = ZeroPadding2D((3, 3))(inpt)

        # conv1
        x = self.Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

        # conv2_x

        x = self.identity_Block(x, nb_filter=64, kernel_size=(3, 3))
        x = self.identity_Block(x, nb_filter=64, kernel_size=(3, 3))
        x = self.identity_Block(x, nb_filter=64, kernel_size=(3, 3))

        # conv3_x

        x = self.identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = self.identity_Block(x, nb_filter=128, kernel_size=(3, 3))
        x = self.identity_Block(x, nb_filter=128, kernel_size=(3, 3))
        x = self.identity_Block(x, nb_filter=128, kernel_size=(3, 3))

        # conv4_x

        x = self.identity_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = self.identity_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = self.identity_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = self.identity_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = self.identity_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = self.identity_Block(x, nb_filter=256, kernel_size=(3, 3))

        # conv5_x

        x = self.identity_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = self.identity_Block(x, nb_filter=512, kernel_size=(3, 3))
        x = self.identity_Block(x, nb_filter=512, kernel_size=(3, 3))


        x = AveragePooling2D(pool_size=(7, 7))(x)

        x = Dropout(dp)(x)

        x = Flatten()(x)
        x = Dense(classes, activation='softmax')(x)


        model = Model(inputs=inpt, outputs=x)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.save("yes")
        return model

    def res50(self, shapes, classes):
        base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=shapes,
                     pooling=None, classes=classes)
        for layer in base_model.layers:
            layer.trainable = False
        x = base_model.output
        x = Dropout(0.5)(x)
        x = Flatten()(x)
        predictions = Dense(classes, activation='softmax', kernel_regularizer=regularizers.l1(0.01))(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        init_lr = 0.001
        opt = SGD(lr=init_lr)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        plot_model(model, to_file='resnet50_model.png', show_shapes=True)
        return model


