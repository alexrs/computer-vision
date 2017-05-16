"""
This file encapsulates the methods used to enhance
the radiographs

See:
    https://www.researchgate.net/publication/235641059_Digital_Radiographic_Image_Enhancement_for_Improved_Visualization
    http://www.ijmlc.org/papers/133-I307.pdf
    http://www.iaescore.com/journals/index.php/IJECE/article/view/5305/5058
    http://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html
    http://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Gradient_Sobel_Laplacian_Derivatives_Edge_Detection.php

Authors: Alejandro Rodriguez, Fernando Collado
"""
import cv2

class Enhancement(object):
    """
    This class represent a radiograph
    """

    @staticmethod
    def fastNlMeansDenoising(img):
        """

        """
        return cv2.fastNlMeansDenoising(img, 10, 10, 7, 21)

    @staticmethod
    def equalizeHist(img):
        """

        """
        return  cv2.equalizeHist(img)

    @staticmethod
    def bilateral_filter(img):
        """
        """
        return cv2.bilateralFilter(img, 9, 175, 175)

    @staticmethod
    def clahe(img):
        """
        """
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        return clahe_obj.apply(img)

    @staticmethod
    def laplacian(img):
        """
        """
        return cv2.Laplacian(img, cv2.CV_64F)

    @staticmethod
    def gaussian_pyramid(img, levels):
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

    @staticmethod
    def closing(img):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def opening(img):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    @staticmethod
    def top_hat(img):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
        return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

    @staticmethod
    def black_hat(img):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

    @staticmethod
    def morfological_gradient(img):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    @staticmethod
    def sobel(img):
        """

        See:
            http://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Gradient_Sobel_Laplacian_Derivatives_Edge_Detection.php
        """
        # remove noise
        img = cv2.GaussianBlur(img, (3,3), 0)
        # convolute with proper kernels
        laplacian = cv2.Laplacian(img,cv2.CV_64F)
        sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
        sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y

        abs_x = cv2.convertScaleAbs(sobelx)
        abs_y = cv2.convertScaleAbs(sobely)
        return cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)