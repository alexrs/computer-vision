"""
Authors: Alejandro Rodriguez, Fernando Collado
file that contains the logic needed to load the data
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

    def get_landmarks(self, incisor, mirrored=True):
        """
        creates an array that contains all the landmarks of a
        directory.

        Params:
            incisor - index of the incisor we want to segmentate

        TODO: Add mirrored landmarks
        """
        folders = ["original"]
        if mirrored:
            folders.append("mirrored")

        # If mirror, the incisor index changes
        mirror_map = {1:4, 2:3, 3:2, 4:1, 5:8, 6:7, 7:6, 8:5}

        # create the array
        shapes = []
        for folder in folders:
            # change the index of the incisor if mirrored
            if folder == "mirrored":
                incisor = mirror_map[incisor]
            # get the filenames that matches the extension
            directory = self.path + "Landmarks/{}/".format(folder)
            # load the landmarks of a given incisor
            filenames = fnmatch.filter(os.listdir(directory), '*-{}.txt'.format(str(incisor)))
            # iterate over the files to create a matrix containing the
            # landmarks
            for filename in filenames:
                file_in = directory + "/" + filename
                landmark = np.loadtxt(file_in, dtype=float)
                shapes.append(Shape.from_list(landmark))

        return shapes

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

