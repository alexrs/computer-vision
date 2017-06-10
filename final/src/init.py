"""
Authors: Alejandro Rodriguez, Fernando Collado

See:
    http://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
    http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_mouse_handling/py_mouse_handling.html
    https://media.readthedocs.org/pdf/opencv-python-tutroals/latest/opencv-python-tutroals.pdf

"""
import cv2
import cv2.cv as cv
import numpy as np
from shape import Shape
from utils import Utils

IMG_WIDTH = 1200
IMG_HEIGHT = 800

class Init(object):
    """
    This class handles the manual and automatic initialization
    for the location of the incisor on the image
    """

    def __init__(self, shape, img, incisive, auto=False):
        """
        Get the shape, the image and if the initialisation is automatic or not
        (By default, it is manual, as this method will be implemented first)
        """

        self.tooth = []
        self.tmpTooth = []
        self.dragging = False
        self.start_point = (img.shape[0]/2, img.shape[1]/2)
        self._initial_fit = None
        if auto:
            self._init_auto(shape, img, incisive)
        else:
            self._init_manual(shape, img)

    def get_initial_fit(self):
        """
        returns the initial fit for the picture
        """
        return self._initial_fit

    def _init_auto(self, shape, img, incisive):
        """
        TODO
        determines the initial fit automatically
        """
        # Cascade path and filename
        cascade_path = "../ProjectData/_Auto/cascade_files/"
        cascade_file = "30_grey_teeth.xml"
        teeth_cascade = cv2.CascadeClassifier(cascade_path+cascade_file)

        # For efficiency's sake, explore a smaller chunk than the original image
        # that is guaranteed to have the incisive teeth in it
        img_t = img[600:1400, 1000:2000]

        # With these params it works the best (img, scale factor, numNeigh)
        teeth = teeth_cascade.detectMultiScale(img_t, 2.3, 150)
        teeth_t = teeth
        # Checking if there are rectangles within rectangles
        if len(teeth) > 1:
            for i, (x, y, w, h) in enumerate(teeth):
                for j, (x1, y1, w1, h1) in enumerate(teeth_t):
                    if i != j and x1 <= x and y1 <= y and \
                        x1+w1 <= x+w and y1+h1 <= y+h:
                        teeth_t[j] = [-1, -1, -1, -1]

        # Obtain rectangles larger than 100x100
        rects = []
        for x, y, w, h in teeth:
            if w > 100 and h > 100:
                rects.append([x, y, w, h])

        # Correct previous img crop
        x_cen = 1000
        y_cen = 600
        x, y, width, height = rects[0]
        width = width/4

        if incisive < 4:
            y_cen += y + height/6
            x_cen += x + (incisive-1) * width + width/2
        else:
            y_cen += y + height/1.2
            x_cen += x + (incisive-5) * width + width/2

        # If visualization is required
        #cv2.rectangle(img, (rectangle[0]+1000, rectangle[1]+600),
        #   (rectangle[0]+rectangle[2]+1000,rectangle[1]+rectangle[3]+600),
        #   (255,0,0), 2)

        # reshape image to fit in the screen
        orig_h = img.shape[0]
        img, scale = Utils.resize(img, IMG_WIDTH, IMG_HEIGHT)
        new_h = img.shape[0]
        tmp = np.array(img)

        # model points to image coordinates
        points = shape.data()
        min_x = abs(points[:, 0].min())
        min_y = abs(points[:, 1].min())
        points = [((point[0]+min_x)*scale*new_h+x_cen*scale, 
                    (point[1]+min_y)*scale*new_h+y_cen*scale) for point in points]


        pimg = np.array([(int(p[0]), int(p[1])) for p in points])

        self.tooth = points

        cv2.polylines(img, [pimg], True, (125, 255, 0), 2)


        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        self._initial_fit = Shape(np.array([[point[0], point[1]] for point in self.tooth]))


    def _init_manual(self, shape, img):
        """
        determines the initial fit manually, dragging the shape
        """

        # reshape image to fit in the screen
        orig_h = img.shape[0]
        img, scale = Utils.resize(img, IMG_WIDTH, IMG_HEIGHT)
        new_h = img.shape[0]
        tmp = np.array(img)

        # model points to image coordinates
        points = shape.data()
        min_x = abs(points[:, 0].min())
        min_y = abs(points[:, 1].min())
        points = [((point[0]+min_x)*scale, (point[1]+min_y)*scale) for point in points]
        self.tooth = points
        pimg = np.array([(int(p[0]*new_h), int(p[1]*new_h)) for p in points])
        cv2.polylines(img, [pimg], True, (125, 255, 0), 2)


        cv2.imshow('img', img)
        cv.SetMouseCallback('img', self._drag, tmp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        self._initial_fit = Shape(np.array([[point[0]*orig_h, point[1]*orig_h] for point in self.tooth]))


    def _drag(self, ev, x, y, flags, img):
        """
        allows user to drag a shape
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


    def _move(self, x, y, img):
        """
        move the shape over the image
        """
        height = img.shape[0]
        tmp = np.array(img)
        dx = (x-self.start_point[0])/float(height)
        dy = (y-self.start_point[1])/float(height)

        points = [(p[0]+dx, p[1]+dy) for p in self.tooth]
        self.tmpTooth = points

        pimg = np.array([(int(p[0]*height), int(p[1]*height)) for p in points])
        cv2.polylines(tmp, [pimg], True, (125, 255, 0), 2)
        cv2.imshow('img', tmp)
