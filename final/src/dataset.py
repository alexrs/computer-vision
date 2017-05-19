"""
Authors: Alejandro Rodriguez, Fernando Collado
file that contains the logic needed to load the data
"""
import os
import fnmatch
import numpy as np
from shape import Shape
import cv2
import re


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

    def load(self, incisor):
        """Collects all example landmark models for the incisor with the given number.

        Args:
            incisor_nr : identifier of tooth

        Returns:
            A list containing all landmark models.

        """
        directory = self.path + "Landmarks/original/"
        files = sorted(fnmatch.filter(os.listdir(directory), "*-{}.txt".format
            (str(incisor))), key=lambda x: int(re.search('[0-9]+', x).group())
            )
        shapes = []
        for filename in files:
            shapes.append(self.get_landmarks(directory + filename))
        return shapes

    def load_mirrored(self, incisor):
        """
        """
        original = self.load(incisor)
        mirror_map = {1: 4, 2: 3, 3: 2, 4: 1, 5: 8, 6: 7, 7: 6, 8: 5}
        mirrored = [shape.mirror_y() for shape in self.load(mirror_map[incisor])]
        return original + mirrored

    def get_landmarks(self, file_in):
        """
        """
        landmark = np.loadtxt(file_in, dtype=float)
        return Shape.from_list_file(landmark)

    def get_images(self):
        """
        get_images loads the radiographs and
        returns a numpy array containing the images
        """
        directory = self.path + "Radiographs/"
        filenames = fnmatch.filter(os.listdir(directory), '*.tif')
        print filenames
        images = []
        for fname in filenames:
            img = cv2.imread(directory + "/" + fname, 0)
            images.append(img)

        return np.array(images)
