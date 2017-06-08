"""
Authors: Alejandro Rodriguez, Fernando Collado
"""
import os
import cv2
import numpy as np

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

    @staticmethod
    def segmentate(test_img, X):
        h, w = test_img.shape
        img = np.zeros((h, w), np.int8)
        mask = np.array([X.data()], dtype=np.int32)
        cv2.fillPoly(img, [mask], 255)
        mask_img = cv2.inRange(img, 1, 255)
        segmented = cv2.bitwise_and(test_img, test_img, mask=mask_img)
        return segmented

    @staticmethod
    def to_binary(img):
        """

        """
        h, w = img.shape
        for i in xrange(h):
            for j in xrange(w):
                if img[i][j] > 0:
                    img[i][j] = 1 #Setting the skin tone to be White
                else:
                    img[i][j] = 0 #else setting it to zero.
        return img


