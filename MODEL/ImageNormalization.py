from keras.preprocessing.image import ImageDataGenerator
import os
import random
import numpy

class ImageNormalization():
    def __init__(self, training_set):
        self.training_set = training_set


    def Normalization(self):
        if not os.path.exists(self.training_set + '/' + 'formal'):
            os.mkdir(self.training_set + '/' + 'formal')
        batch_size = 1
        train_datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)

        allDir = os.listdir(self.training_set)
        number=[];
        for dir in allDir:
            sonDirName = os.path.join(self.training_set, dir)
            # print('read class')
            if os.path.isdir(sonDirName):
                pathDir = os.listdir(sonDirName)
                #print(dir)
                filenumber = len(pathDir)
            number.append(filenumber)
        for dir in allDir:
            sonDirName = os.path.join(self.training_set, dir)
            # print('read class')
            i = 1
            if os.path.isdir(sonDirName):
                pathDir = os.listdir(sonDirName)
                # print(dir)
                filenumber = len(pathDir)
                print (filenumber)

            i=0
            if dir == 'formal':
                break
            else:
                if not os.path.exists(self.training_set + '/' + 'formal' + '/' + dir):
                    os.mkdir(self.training_set + '/' + 'formal' + '/' + dir)
                for batch in train_datagen.flow_from_directory(self.training_set,
                                                        classes = [str(dir)],
                                                        batch_size=batch_size,
                                                        save_to_dir=self.training_set + '/' + 'formal' + '/' + dir,
                                                        save_prefix='ge',
                                                        save_format='tif',
                                                        ):
                    i += 1
                    if i >= max(number):
                        break
