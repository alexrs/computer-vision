"""
Authors: Alejandro Rodriguez, Fernando Collado

See: 
    http://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
    http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_mouse_handling/py_mouse_handling.html

"""

import cv2.cv as cv
import cv2
import numpy as np
from shape import Shape

IMG_WIDTH = 1200
IMG_HEIGHT = 800


class Init(object):
    """
    This class handles the manual and automatic initialization
    for the location of the incisor on the image
    """

    def __init__(self, shape, img, auto=False):
        """
        Get the shape, the image and if the initialisation is automatic or not
        (By default, it is manual, as this method will be implemented first)
        """

        self.tooth = []
        self.tmpTooth = []
        self.dragging = False
        self.start_point = (0, 0)
        self.init(shape, img)


    def init(self, shape, img):
        """

        """

        orig_h = img.shape[0]
        img, scale = self.resize(img, IMG_WIDTH, IMG_HEIGHT)
        new_h = img.shape[0]
        canvasing = np.array(img)

        # transform model points to image coord
        points = shape.data()
        min_x = abs(points[:, 0].min())
        min_y = abs(points[:, 1].min())
        points = [((point[0]+min_x)*scale, (point[1]+min_y)*scale) for point in points]
        self.tooth = points
        pimg = np.array([(int(p[0]*new_h), int(p[1]*new_h)) for p in points])
        cv2.polylines(img, [pimg], True, (0, 255, 0))


        cv2.imshow('img', img)
        cv.SetMouseCallback('img', self._drag, canvasing)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        centroid = np.mean(self.tooth, axis=0)
        return Shape(np.array([[point[0]*orig_h, point[1]*orig_h] for point in self.tooth]))


    def _drag(self, ev, x, y, flags, img):
        """
        """
        if ev == cv.CV_EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start_point = (x, y)
        elif ev == cv.CV_EVENT_LBUTTONUP:
            self.tooth = self.tmpTooth
            self.dragging = False
        elif ev == cv.CV_EVENT_MOUSEMOVE:
            if self.dragging and self.tooth != []:
                self._move(x, y, img)

    def resize(self, img, new_w, new_h):
        """
        Resize the image and return the new image, and the scale factor

        https://enumap.wordpress.com/2014/07/06/python-opencv-resize-image-by-width/
        """
        #find minimum scale to fit image on screen
        h, w = img.shape
        scale = min(float(new_w) / w, float(new_h) / h)
        return cv2.resize(img, (int(w * scale), int(h * scale))), scale


    def _move(self, x, y, img):
        """
        """
        height = img.shape[0]
        tmp = np.array(img)
        dx = (x-self.start_point[0])/float(height)
        dy = (y-self.start_point[1])/float(height)

        points = [(p[0]+dx, p[1]+dy) for p in self.tooth]
        self.tmpTooth = points

        pimg = np.array([(int(p[0]*height), int(p[1]*height)) for p in points])
        cv2.polylines(tmp, [pimg], True, (0, 255, 0))
        cv2.imshow('img', tmp)

    