import os
import numpy
import random, shutil

class ReadImage():
    def __init__(self, file_path, target_path):
        self.file_path = file_path
        self.target_path = target_path


    def makecopy(self):
        if not os.path.exists(self.target_path):
            try:
                print('making copy')
                shutil.copytree(self.file_path, self.target_path)
            except shutil.Error as e:
                for self.file_path, self.target_path, msg in e.args[0]:
                    print(self.file_path, self.target_path, msg)
        else:
            print('path exist')

    def move_image(self):
        allDir = os.listdir(self.target_path)
        if os.path.exists(self.target_path + '/' + 'Release'):
            print('removing Release path')
        else:
            os.mkdir(self.target_path + '/' + 'Release')

        for dir in allDir:
            # 5 Class
            sonDirName = os.path.join(self.target_path, dir)
            #print('read class')
            i=0
            sonDir=os.listdir(sonDirName)
            for subdir in sonDir:
                imDirName = os.path.join(sonDirName, subdir)
                #print(imDirName)
                imDir=os.listdir(imDirName)
                for im in imDir:
                    if not os.path.exists(self.target_path + '/' + 'Release' + '/' + dir +'_copy'):
                        os.mkdir(self.target_path + '/' + 'Release' + '/' + dir +'_copy')
                    imname=os.path.join(imDirName, im)
                    #print(imname)
                    tardir = self.target_path + '/' + 'Release' + '/' + dir +'_copy' + '/' + str(i)
                    #print(tardir)
                    shutil.copy(imname, tardir)
                    i += 1