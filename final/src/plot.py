"""
Authors: Alejandro Rodriguez, Fernando Collado
http://docs.opencv.org/3.0-beta/modules/imgproc/doc/drawing_functions.html
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
from shape import Shape
import colorsys
import time
from utils import Utils

HEIGHT = 800
WIDTH = 1200


class Plot(object):
    """
    Whatever you want to plot, you can find it here
    """

    @staticmethod
    def perf(perf):
        ind = np.arange(len(perf))
        width = 0.35       # the width of the bars
        plt.bar(ind, perf, width, color='g')
        plt.axhline(np.mean(perf))
        plt.xlabel('Incisor')
        plt.ylabel('Accuracy')
        plt.axis([0, 7, 0, 1])
        plt.grid(True)
        plt.show()

    @staticmethod
    def variance(eig_vals):
        """
        plot the variance
        """
        tot = sum(eig_vals)
        var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
        cum_var_exp = np.cumsum(var_exp)
        print cum_var_exp
        with plt.style.context('seaborn-whitegrid'):
            plt.figure(figsize=(6, 4))
            plt.bar(range(80), var_exp, alpha=0.5, align='center',
                    label='individual explained variance')
            plt.step(range(80), cum_var_exp, where='mid',
                     label='cumulative explained variance')
            plt.ylabel('Explained variance ratio')
            plt.xlabel('Principal components')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.show()

    @staticmethod
    def image(img):
        """
        show an image
        """
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
        plt.show()

    @staticmethod
    def active_shape_model(active_shape_model):
        """
        plot Active Shape Models with +-3 std
        """
        mean_shape = active_shape_model.mean_shape().collapse()
        pc_modes = active_shape_model.pca().pc_modes()

        for i in range(8): # iterate over the pca dimensions
            shapes = [Shape.from_list_consecutive(mean_shape-j*pc_modes[:, i]) for j in range(-3, 4)]
            Plot.shapes(shapes)

    @staticmethod
    def shapes(shapes):
        """
        """
        margin = 10

        shapes = [shape.normalize().scale(1000) for shape in shapes]
        colors = Plot._get_colors(len(shapes))

        max_x = int(max([shape.data()[:, 0].max() for shape in shapes]))
        max_y = int(max([shape.data()[:, 1].max() for shape in shapes]))
        min_x = int(min([shape.data()[:, 0].min() for shape in shapes]))
        min_y = int(min([shape.data()[:, 1].min() for shape in shapes]))

        img = np.ones((max_y-min_y+20, max_x-min_x+20, 3), np.uint8)*255 # white image
        for i, shape in enumerate(shapes):
            points = shape.data().astype(int)
            for j in range(len(points)):
                cv2.line(img, (points[j, 0] - min_x + margin, points[j, 1] - min_y + margin),
                        (points[(j + 1) % 40, 0]-min_x + margin, points[(j + 1) % 40, 1]-min_y + margin),
                        colors[i], thickness=1, lineType=cv2.CV_AA)

        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def approximated_shape(shapes, img, show=True, wait=True):
        """
        """
        img = img.copy()

        colors = Plot._get_colors(len(shapes))
        for i, shape in enumerate(shapes):
            points = shape.data().astype(int)
            for j, p in enumerate(points):
                cv2.line(img, (p[0], p[1]),
                        (points[(j + 1)%40, 0], points[(j + 1)%40, 1]),
                        colors[i], 2)

        if show:
            img, _ = Utils.resize(img, WIDTH, HEIGHT)
            cv2.imshow("", img)
            if wait:
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                cv2.waitKey(1)

    @staticmethod
    def _get_colors(num_colors):
        """
        https://www.quora.com/How-do-I-generate-n-visually-distinct-RGB-colours-in-Python/answer/Reed-Oei?srid=olmQ
        """
        max_value = 16581375 #255**3
        interval = int(max_value / num_colors)
        colors = [hex(i)[2:].zfill(6) for i in range(0, max_value, interval)]

        return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]
