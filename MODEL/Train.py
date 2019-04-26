from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend
from MODEL.Models import *
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import argparse
import sys

class Train():
    def __init__(self, train_data, train_labels, test_data, test_labels, class_list):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.class_list = class_list

    def train(self, model, input_data, input_labels, val_data, val_labels):
        epoches = 50
        models = model.fit(input_data, input_labels, validation_data=(val_data, val_labels), epochs=epoches, batch_size=32)
        print('[INFO] Evaluating model')
        predictions = model.predict(val_data, batch_size=32)
        print(classification_report(val_labels.argmax(axis=1), predictions.argmax(axis=1),
                                    target_names=['DNB', 'GN', 'GNB_3_SP', 'PDNB', 'UDNB']))

        N = np.arange(0, epoches)
        plt.style.use('ggplot')
        plt.figure()
        plt.plot(N, models.history['loss'], label='train_loss')
        plt.plot(N, models.history['val_loss'], label='val_loss')
        plt.plot(N, models.history['acc'], label='train_acc')
        plt.plot(N, models.history['val_acc'], label='val_acc')
        plt.title('Training loss and Accuracy')
        plt.xlabel('epoch #')
        plt.ylabel('loss/accuracy')
        plt.legend()
        plt.savefig('plot.png')
        #model.save(sys.args['model'])
