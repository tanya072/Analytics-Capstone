import sys
import os
from MODEL.ReadImage import *
from MODEL.ImagePartition import *
from MODEL.ImageGeneration import *
from MODEL.ImageNormalization import *
from MODEL.Models import *
from MODEL.Train import *
from sklearn.preprocessing import LabelBinarizer


class APPSTART():
    def __init__(self):

        self.file_path = 'C:/Users/bobby/Documents/Data/LAST_VERSION_15-4-2017'
        self.target_path = 'C:/Users/bobby/Documents/Data/copy_data'
        self.ReadImage = ReadImage(self.file_path, self.target_path)
        self.released_image = 'C:/Users/bobby/Documents/Data/copy_data/Release'
        self.ImagePartition = ImagePartition(self.released_image,self.target_path)
        self.training_set = 'C:/Users/bobby/Documents/Data/copy_data/train_set'
        self.test_set = 'C:/Users/bobby/Documents/Data/copy_data/test_set'
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
        self.model1 = self.Models.seq_setting(self.norm_size*self.norm_size*3)
        self.model2 = self.Models.Res_setting(self.norm_size, self.norm_size, 3, self.classes)

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
        self.Train.train(self.model2, self.train_data, self.train_labels, self.test_data, self.test_labels)




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
