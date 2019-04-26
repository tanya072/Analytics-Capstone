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
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils



class Models():
    def __init__(self, classes, train_data, train_labels, class_list):
        self.classes = classes
        self.train_data = train_data
        self.train_labels = train_labels
        self.class_list = class_list

    def seq_setting(self, norm_size):
        model = Sequential()
        #model.add(Dropout(0.5))
        model.add(Dense(512, input_shape=(norm_size,), activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(5, activation='softmax'))

        init_lr = 0.01
        opt = SGD(lr=init_lr)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    @staticmethod
    def inc_setting():
        base_model = InceptionV3(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(1, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        for layer in base_model.layers:
            layer.trainable = False
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        return model
