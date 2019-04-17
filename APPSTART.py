import sys
import os
from MODEL.ReadImage import *



class APPSTART():
    def __init__(self, object):
        self.object = object
        self.file_path= '/home/bobby/Documents/cancer_dataset/LAST_VERSION_15-4-2017'
        self.target_path= '/home/bobby/Documents/cancer_dataset/copy_data'
        self.ReadImage=ReadImage(self.file_path, self.target_path)
    def run(self):
        print('Welcome')
        self.ReadImage.makecopy()
        self.ReadImage.move_image()

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
            os._exit(0)
