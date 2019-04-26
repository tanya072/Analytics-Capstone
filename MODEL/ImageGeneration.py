from keras.preprocessing.image import ImageDataGenerator
import os
import random
import numpy
class ImageGeneration():
    def __init__(self, training_set):
        self.training_set = training_set


    def Generation(self):
        batch_size = 1
        train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=45, height_shift_range=0.2,
                                           width_shift_range=0.2, horizontal_flip=True,
                                           fill_mode='reflect')

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
                #picknumber = int(max(number) - filenumber)
                picknumber = int(600 - filenumber)
                print (max(number))
                print (filenumber)
                print (picknumber)
            i=0
            if picknumber == 0:
                break
            else:
                for batch in train_datagen.flow_from_directory(self.training_set,
                                                        classes = [str(dir)],
                                                        batch_size=batch_size,
                                                        save_to_dir=self.training_set + '/' + dir,
                                                        save_prefix='ge',
                                                        save_format='tif',
                                                        ):
                    i += 1
                    if i >= picknumber:
                        break
