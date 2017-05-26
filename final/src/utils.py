"""
Authors: Alejandro Rodriguez, Fernando Collado
"""
import os
import cv2

K = 10
M = 15

class Utils(object):
    """
    The code that does not fit anywhere else
    """

    @staticmethod
    def create_dir(dirname):
        """
        Creates a directory if it does not exists
        """
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    @staticmethod
    def resize(img, new_w, new_h):
        """
        Resize the image and return the new image, and the scale factor

        https://enumap.wordpress.com/2014/07/06/python-opencv-resize-image-by-width/
        """
        h, w = img.shape
        scale = min(float(new_w) / w, float(new_h) / h)
        return cv2.resize(img, (int(w * scale), int(h * scale))), scale

