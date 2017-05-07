import os
import fnmatch
import numpy as np
from shape import Shape

class Dataset:

    def __init__(self, path):
        self.path = path
        
    def get_landmarks(self):
        '''
        load_landmarks creates an array that contains all the landmarks of a
        directory.
        @return np.array, shape=(num of files, num of landmarks in each file=80)
        '''
        #get the flenames that matches the extension
        directory = self.path + "Landmarks/original/"
        filenames = fnmatch.filter(os.listdir(directory), '*.txt')
        # create the array
        X = []
        # iterate over the files to create a matrix containing the
        # landmarks
        for filename in filenames:
            file_in = directory + "/" + filename
            lands = np.loadtxt(file_in, dtype=float)
            X.append(self.load_landmark(lands))
            
        return X

    def load_landmark(self, landmark):
        x = []
        y = []
        for i, line in enumerate(landmark):
            if i % 2 == 0:
                x.append(float(line))
            else:
                y.append(float(line))
        return Shape.from_points(x, y)
        
    def get_images(self):
        directory = self.path + "Radiographs/"
        filenames = fnmatch.filter(os.listdir(directory),'*.tif')

