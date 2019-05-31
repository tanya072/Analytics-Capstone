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
from keras.models import Model
import matplotlib.pyplot as plt
from keras.preprocessing import image
import argparse
from keras.applications.imagenet_utils import decode_predictions


class Prediction():
    def __init__(self, predict_data, norm_size, lb):
        self.predict_data = predict_data
        self.norm_size = norm_size
        self.lb = lb

    def load_data(self, target_set, target_data, target_labels, norm_size):
        print('Loading Image')
        imagepaths = sorted(list(paths.list_images(target_set)))
        random.seed(43)
        random.shuffle(imagepaths)
        for imagePath in imagepaths:
            image = cv2.imread(imagePath)
            # image = cv2.resize(image, (norm_size, norm_size)).flatten()
            image = cv2.resize(image, (norm_size, norm_size))
            image = img_to_array(image)
            # image = np.expand_dims(image, axis=0)
            # image = image.reshape((1,) + image.shape)
            target_data.append(image)
            label = str(imagePath.split(os.path.sep)[-2])
            target_labels.append(label)

    @staticmethod
    def data_normalization(input_data):
        return np.array(input_data, dtype="float") / 255.0

    def prediction(self, target_set, norm_size, model):
        images = cv2.imread(target_set)
        # image = cv2.resize(image, (norm_size, norm_size)).flatten()
        image = cv2.resize(images, (norm_size, norm_size))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = np.array(image, dtype="float") / 255.0
        #image = preprocess_input(image)
        #image = image.reshape((1,) + image.shape)
        preds = model.predict(image, verbose=1)
        print(preds)
        preds = self.lb.inverse_transform(preds)
        print(preds)

