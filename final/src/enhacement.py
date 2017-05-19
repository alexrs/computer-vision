"""
Authors: Alejandro Rodriguez, Fernando Collado

This file encapsulates the methods used to enhance
the radiographs

See:
    https://www.researchgate.net/publication/235641059_Digital_Radiographic_Image_Enhancement_for_Improved_Visualization
    http://www.ijmlc.org/papers/133-I307.pdf
    http://www.iaescore.com/journals/index.php/IJECE/article/view/5305/5058
    http://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html
    http://ieeexplore.ieee.org/document/6805749/?part=1
    http://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Gradient_Sobel_Laplacian_Derivatives_Edge_Detection.php
"""
import cv2
from utils import Utils


class Enhancement(object):
    """
    Enhacement algorithms to improve the radiographs
    """

    @staticmethod
    def enhance(img, index, save=False, path="../ProjectData/_Data/Radiographs/Enhanced"):
        """
        Returns the enhaced image
        """
        # 1. Denoise the image
        img = Enhancement.fastNlMeansDenoising(img)
        # Bilateral filter is faster, but the results are worse (TODO: Tunning parameters)
        # img = Enhancement.bilateral_filter(img)
        # 2. Apply top hat.
        top_hat_img = Enhancement.top_hat(img)
        # 3. Apply black hat
        black_hat_img = Enhancement.black_hat(img)
        # 4. Add the returned top hat image to the original image
        img = img + top_hat_img
        # 5.  Substract the black hat image to the original image
        img = img - black_hat_img
        # 6. Apply CLAHE to enhance the contrast
        # (https://en.wikipedia.org/wiki/Adaptive_histogram_equalization)
        img = Enhancement.clahe(img)
        # 7. Apply sobel to detect the edges (https://en.wikipedia.org/wiki/Sobel_operator)
        img = Enhancement.sobel(img)

        if save:
            Utils.create_dir(path)
            cv2.imwrite("{}/{}.tif".format(path, str(index + 1).zfill(2)), img)
        return img


    @staticmethod
    def fastNlMeansDenoising(img):
        """
        Reduce noise in the image
        """
        return cv2.fastNlMeansDenoising(img, 10, 10, 7, 21)

    @staticmethod
    def equalizeHist(img):
        """
        equalize the histogram of the image
        """
        return  cv2.equalizeHist(img)

    @staticmethod
    def bilateral_filter(img):
        """
        """
        return cv2.bilateralFilter(img, 15, 80, 80)

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

        TODO: Think if we really need it? - Do the complete implementation if enough time
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
        """
        Useful to remove small holes (dark regions).
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def opening(img):
        """
        Useful for removing small objects
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    @staticmethod
    def top_hat(img):
        """
        It is the difference between an input image and its opening
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

    @staticmethod
    def black_hat(img):
        """
        It is the difference between the closing and its input image
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

    @staticmethod
    def morfological_gradient(img):
        """
        It is the difference between the dilation and the erosion of an image
        It is useful for finding the outline of an object
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    @staticmethod
    def sobel(img):
        """

        See:
            http://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Gradient_Sobel_Laplacian_Derivatives_Edge_Detection.php
        """
        # remove noise
        img = cv2.GaussianBlur(img, (3, 3), 0)
        # convolute with proper kernels
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # x
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # y

        abs_x = cv2.convertScaleAbs(sobelx)
        abs_y = cv2.convertScaleAbs(sobely)
        return cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
