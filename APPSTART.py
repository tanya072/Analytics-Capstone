import sys
import os
from MODEL.ReadImage import *
from MODEL.ImagePartition import *
from MODEL.ImageGeneration import *

class APPSTART():
    def __init__(self, object):
        self.object = object
        self.file_path = '/home/bobby/Documents/cancer_dataset/LAST_VERSION_15-4-2017'
        self.target_path = '/home/bobby/Documents/cancer_dataset/copy_data'
        self.ReadImage = ReadImage(self.file_path, self.target_path)
        self.released_image = '/home/bobby/Documents/cancer_dataset/copy_data/Release'
        self.ImagePartition = ImagePartition(self.released_image,self.target_path)
        self.training_set = '/home/bobby/Documents/cancer_dataset/copy_data/train_set'
        self.ImageGeneration = ImageGeneration(self.training_set)
    def run(self):
        self.ReadImage.makecopy()
        self.ReadImage.move_image()
        self.ImagePartition.Partition('0.2')
        self.ImageGeneration.Generation()

if __name__ == '__main__':
    try:
        APP = APPSTART(object)
        APP.run()

    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(1)
        except SystemExit:
            print("Exception when exit python")
            os.exit(0)
