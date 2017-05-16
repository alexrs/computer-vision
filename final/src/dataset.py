"""
Authors: Alejandro Rodriguez, Fernando Collado
file that contains the logig needed to load the data
"""
import os
import fnmatch
import numpy as np
from shape import Shape
import cv2

class Dataset(object):
    '''
    This class implements the needed methods to load the landmarks
    and images
    '''

    def __init__(self, path="../ProjectData/_Data/"):
        """
        path -- path where the dataset is located
        """
        self.path = path

    def get_landmarks(self, incisor, mirrored=False):
        """
        creates an array that contains all the landmarks of a
        directory.

        Params:
            incisor - index of the incisor we want to segmentate

        TODO: Add mirrored landmarks
        """
        # get the filenames that matches the extension
        directory = self.path + "Landmarks/original/"
        # load the landmarks of a given incisor
        filenames = fnmatch.filter(os.listdir(directory), '*-{}.txt'.format(str(incisor)))
        # create the array
        landmarks = []
        # iterate over the files to create a matrix containing the
        # landmarks
        for filename in filenames:
            file_in = directory + "/" + filename
            lands = np.loadtxt(file_in, dtype=float)
            landmarks.append(self._load_landmark(lands))

        return landmarks

    def get_images(self):
        """
        get_images loads the radiographs and
        returns a numpy array containing the images
        """
        directory = self.path + "Radiographs/"
        filenames = fnmatch.filter(os.listdir(directory), '*.tif')
        images = []
        for fname in filenames:
            img = cv2.imread(directory + "/" + fname, 0)
            images.append(img)

        return np.array(images)

    def _load_landmark(self, landmark):
        """
        _load_landmark returns a Shape given a landmark
        """
        x = []
        y = []
        for i, line in enumerate(landmark):
            if i % 2 == 0:
                x.append(float(line))
            else:
                y.append(float(line))
        return Shape.from_points(x, y)

