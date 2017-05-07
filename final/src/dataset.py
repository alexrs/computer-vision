import os
import fnmatch
import numpy as np
from shape import Shape
import cv2

class Dataset:
    '''
    This class implements the needed methods to load the landmarks
    and images
    '''

    def __init__(self, path):
        """
        path -- path where the dataset is located
        """
        self.path = path
        
    def get_landmarks(self):
        """
        creates an array that contains all the landmarks of a
        directory.
        """
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
        images = []
        for fname in filenames:
            img = cv2.imread(directory + "/" + fname, 0)
            images.append(img)
            
        return np.array(images)

