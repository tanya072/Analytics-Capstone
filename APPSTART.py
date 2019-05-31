# -*- coding: utf-8 -*-
import sys
import os
from MODEL.ReadImage import *
from MODEL.ImagePartition import *
from MODEL.ImageGeneration import *
from MODEL.ImageNormalization import *
from MODEL.Models import *
from MODEL.Train import *
from sklearn.preprocessing import LabelBinarizer
from MODEL.Prediction import *

class APPSTART():
    def __init__(self):

        self.file_path = '/home/dylan/文档/data/LAST_VERSION_15-4-2017'
        self.target_path = '/home/dylan/文档/data/copy_data'
        self.ReadImage = ReadImage(self.file_path, self.target_path)
        self.released_image = '/home/dylan/文档/data/copy_data/Release'
        self.ImagePartition = ImagePartition(self.released_image,self.target_path)
        self.training_set = '/home/dylan/文档/data/copy_data/train_set'
        self.test_set = '/home/dylan/文档/data/copy_data/test_set'
        self.predict_data = '/home/dylan/文档/data/test_data/DNB/I (1).tif'
        self.ImageGeneration = ImageGeneration(self.training_set)
        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []
        self.norm_size = 224
        self.lb = LabelBinarizer()
        self.ImageNormalization = ImageNormalization(self.training_set, self.test_set, self.train_data,
                                                     self.train_labels, self.test_data, self.test_labels, self.lb)
        self.classes = 5
        self.class_list = []
        self.Models = Models(self.classes, self.train_data, self.train_labels, self.test_data, self.test_labels,
                             self.class_list)
        self.Train = Train(self.train_data, self.train_labels, self.test_data, self.test_labels, self.class_list)
        self.Prediction = Prediction(self.predict_data, self.norm_size, self.lb)
        #self.model1 = self.Models.seq_setting(self.norm_size*self.norm_size*3)
        self.model2 = self.Models.Res_setting(self.norm_size, self.norm_size, 3, self.classes)
        self.model3 = self.Models.Res_setting(self.norm_size, self.norm_size, 3, self.classes, dp=0, opt=SGD(lr=0.01))
        self.model4 = self.Models.Res_setting(self.norm_size, self.norm_size, 3, self.classes, dp=0.5)
        self.model5 = self.Models.Res_setting(self.norm_size, self.norm_size, 3, self.classes, dp=0.75)
        #self.model6 = self.Models.res50((self.norm_size, self.norm_size, 3), self.classes)
        self.model7 = self.Models.Res_setting(self.norm_size, self.norm_size, 3, self.classes, opt=SGD(lr=0.001))
        self.model8 = self.Models.Res_setting(self.norm_size, self.norm_size, 3, self.classes, opt=RMSprop(lr=0.001))
        self.model9 = self.Models.Res_setting(self.norm_size, self.norm_size, 3, self.classes, opt=Adagrad(lr=0.001))
        self.model10 = self.Models.Res_setting(self.norm_size, self.norm_size, 3, self.classes, opt=Adam(lr=0.001))
        self.model11 = self.Models.Res_setting(self.norm_size, self.norm_size, 3, self.classes, opt=SGD(lr=0.005))
        self.model12 = self.Models.Res_setting(self.norm_size, self.norm_size, 3, self.classes, opt=SGD(lr=0.01))
        self.model13 = self.Models.Res_setting(self.norm_size, self.norm_size, 3, self.classes, opt=SGD(lr=0.0001))

    def run(self):
        #self.ReadImage.makecopy()
        #self.ReadImage.move_image()
        #self.ImagePartition.Partition('0.2')
        #self.ImageGeneration.Generation()
        self.ImageNormalization.load_image(self.training_set, self.train_data, self.train_labels, self.norm_size)
        self.ImageNormalization.load_image(self.test_set, self.test_data, self.test_labels, self.norm_size)
        self.train_data = self.ImageNormalization.data_normalization(self.train_data)
        self.train_labels = self.ImageNormalization.label_normalization(self.train_labels)

        self.test_data = self.ImageNormalization.data_normalization(self.test_data)
        self.test_labels = self.ImageNormalization.label_normalization(self.test_labels)
        self.class_list = self.lb.classes_
        #print(self.class_list)
        #self.Train.train(self.model1, self.train_data, self.train_labels, self.test_data, self.test_labels)
        self.Train.train(self.model3, self.train_data, self.train_labels, self.test_data, self.test_labels, 'yy')
        #self.Train.train(self.model6, self.train_data, self.train_labels, self.test_data, self.test_labels, 'RES_50')
        #self.Train.train(self.model3, self.train_data, self.train_labels, self.test_data, self.test_labels, 'dp0.25')
        #self.Train.train(self.model4, self.train_data, self.train_labels, self.test_data, self.test_labels, 'dp0.5')
        #self.Train.train(self.model5, self.train_data, self.train_labels, self.test_data, self.test_labels, 'dp0.75')
        #self.Train.train(self.model7, self.train_data, self.train_labels, self.test_data, self.test_labels, 'dp0_SGD0.001')
        #self.Train.train(self.model8, self.train_data, self.train_labels, self.test_data, self.test_labels, 'dp0_RMSprop')
        #self.Train.train(self.model9, self.train_data, self.train_labels, self.test_data, self.test_labels, 'dp0_Adagrad')
        #self.Train.train(self.model10, self.train_data, self.train_labels, self.test_data, self.test_labels, 'dp0_Adam')
        # self.Train.train(self.model11, self.train_data, self.train_labels, self.test_data, self.test_labels,'dp0_SGD0.005')
        # self.Train.train(self.model12, self.train_data, self.train_labels, self.test_data, self.test_labels,
        #                  'dp0_SGD0.01')
        # self.Train.train(self.model13, self.train_data, self.train_labels, self.test_data, self.test_labels,
        #                  'dp0_SGD0.0001')
        self.Prediction.prediction(self.predict_data, self.norm_size, self.model3)

if __name__ == '__main__':
    try:
        APP = APPSTART()
        APP.run()

    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(1)
        except SystemExit:
            print("Exception when exit python")
            sys.exit(0)
