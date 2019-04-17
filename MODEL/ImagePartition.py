import os
import random
import shutil

class ImagePartition():
    def __init__(self, released_image, target_path):
        self.released_image = released_image
        self.target_path = target_path

    def Partition(self, rate):
        allDir = os.listdir(self.released_image)
        if not os.path.exists(self.target_path + '/' + 'test_set'):
            os.mkdir(self.target_path + '/' + 'test_set')
        if not os.path.exists(self.target_path + '/' + 'train_set'):
            os.mkdir(self.target_path + '/' + 'train_set')
        for dir in allDir:
            if not os.path.exists(self.target_path + '/' + 'test_set' + '/' + dir):
                os.mkdir(self.target_path + '/' + 'test_set' + '/' + dir)
            if not os.path.exists(self.target_path + '/' + 'train_set' + '/' + dir):
                os.mkdir(self.target_path + '/' + 'train_set' + '/' + dir)
            # 5 Class
            sonDirName = os.path.join(self.released_image, dir)
            if os.path.isdir(sonDirName):
                pathDir = os.listdir(sonDirName)
                #print(dir)
                filenumber = len(pathDir)
                picknumber = int(filenumber * float(rate))
                sample = random.sample(pathDir, picknumber)
                #print(sample)
                for name in sample:
                    #print(name)
                    oldDir = self.released_image + '/' + dir + '/' + name
                    newtestDir = self.target_path + '/' + 'test_set' + '/' + dir + '/' + name
                    shutil.move(oldDir, newtestDir)
                for name in pathDir:
                    try:
                        oldDir = self.released_image + '/' + dir + '/' + name
                        newtrainDir = self.target_path + '/' + 'train_set' + '/' + dir + '/' + name
                        shutil.move(oldDir, newtrainDir)
                    except FileNotFoundError:
                        None

