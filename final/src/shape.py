import numpy as np
from scipy import spatial
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2

class Shape:

    def __init__(self, data):
        self._data = data

    @classmethod
    def from_points(cls, x, y):
        if len(x) != len(y):
            raise AssertionError("Lenght should be equal")
        points = []
        for i in range(len(x)):
            points.append([x[i], y[i]])
        return cls(np.array(points))
        
    def mean(self):
        return np.mean(self._data, axis=0)

    def norm(self):
        return np.linalg.norm(self._data, axis=0)

    def shape(self):
        return self._data.shape

    def center(self):
        return Shape(self._data - self.mean())

    def normalize(self):
        return Shape(self._data / self.norm())

    def align(self, other):
        """
        Aligns the current shape (HAS TO BE CENTERED)
        to the other shape (HAS TO BE CENTERED AS WELL) by
        finding a transformation matrix  r by solving the
        least squares solution of the equation
        self*r = other
        :param other: The other shape
        :return: A shape aligned to other
        """
        other_data = other.data()
        cov = np.dot(other_data.T, self._data)
        btb = np.dot(other_data.T, other_data)
        pic = np.linalg.pinv(cov)
        r = np.dot(pic, btb)
        return Shape(np.dot(self._data, r))

    def scale(self, factor):
        return Shape(factor * self._data)

    def translate(self, displacement):
        return Shape(self._data + displacement)

    def collapse(self):
        n, _ = self._data.shape
        return np.reshape(self._data, 2 * n)

    def data(self):
        return self._data

class AlignedShape:

    def __init__(self, shapes,  tol=1e-7, max_iters=10000):
        self._align(shapes, tol, max_iters)
            
    def mean_shape(self):
        """
        Returns the mean shape
        :return: A shape object containing the mean shape
        """
        return self._mean_shape

    def shapes(self):
        return self._aligned_shapes

    def data(self):
        """
        Returns the shapes as a numpy matrix.
        :return: a Nxd*d numpy matrix containing the shapes
        """
        return np.array([shape.collapse() for shape in self._aligned_shapes])

    def _align(self, shapes,  tol, max_iters):
        """
        1. Rotate, scale and translate each shape to align with the
        first shape in the set.
        2. Repeat
        |   2.1. Calculate the mean shape from the aligned shapes
        |   2.2. Normalize the orientation, scale and origin of the
        |   current mean to suitable defaults
        |   2.3. Realign every shape with the current mean
        3. Until the process converges
        """
        self._aligned_shapes = [shape.center() for shape in shapes]
        self._mean_shape = self._aligned_shapes[0].normalize()
        for num_iters in range(max_iters):
            for i in range(len(self._aligned_shapes)):
                self._aligned_shapes[i] = self._aligned_shapes[i].align(self._mean_shape)
            previous_mean_shape = self._mean_shape
            self._mean_shape = Shape(
                np.mean(np.array([shape.data() for shape in self._aligned_shapes])
                        , axis=0)).center().normalize()
            if np.linalg.norm(self._mean_shape.data() - previous_mean_shape.data()) < tol:
                break

