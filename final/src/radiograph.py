"""
This file encapsulates the methods used to enhance
the radiographs

Authors: Alejandro Rodriguez, Fernando Collado
"""
import cv2

class Radiograph(object):
    """
    This class represent a radiograph
    """

    def __init__(self, radiograph):
        self.radiograph = radiograph

    def gaussian_pyramid(self, img, levels):
        """
        gaussian_pyramid creates a gaussian pyramid of an image
        given the number of levels.

        See: http://docs.opencv.org/3.1.0/dc/dff/tutorial_py_pyramids.html
        """
        pyramid = []
        # Append the original image as first layer of the pyramid
        pyramid.append(img)
        tmp = img
        for _ in range(levels):
            tmp = cv2.pyrDown(tmp)
            # append a new layer with a lower res
            pyramid.append(tmp)

        return pyramid
