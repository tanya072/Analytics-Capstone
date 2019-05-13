import os
import random
import numpy
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelBinarizer
from imutils import paths
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import argparse
import random
import cv2
import os
import sys


class ImageNormalization():
    def __init__(self, training_set, test_set, train_data, train_labels, test_data, test_labels, lb):
        self.training_set = training_set
        self.test_set = test_set
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.lb = lb


    @staticmethod
    def load_image(target_set, target_data, target_labels, norm_size):
        print('Loading Image')
        imagepaths = sorted(list(paths.list_images(target_set)))
        random.seed(43)
        random.shuffle(imagepaths)
        for imagePath in imagepaths:
            image = cv2.imread(imagePath)
            #image = cv2.resize(image, (norm_size, norm_size)).flatten()
            image = cv2.resize(image, (norm_size, norm_size))
            image = img_to_array(image)
            #image = np.expand_dims(image, axis=0)
            #image = image.reshape((1,) + image.shape)
            target_data.append(image)
            label = str(imagePath.split(os.path.sep)[-2])
            target_labels.append(label)

    @staticmethod
    def data_normalization(input_data):
        return np.array(input_data, dtype="float") / 255.0

    def label_normalization(self, input_labels):
        #return np.asarray(MultiLabelBinarizer().fit_transform(input_labels)).astype('float32')
        return np.asarray(self.lb.fit_transform(input_labels)).astype('float32')

    @staticmethod
    def get_training_data(input_data, input_labels):
        for i in range(0, 2999):
            image = input_data[i]
            label = input_labels[i]
            yield (image, label)
