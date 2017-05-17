"""
Authors: Alejandro Rodriguez, Fernando Collado
"""
import numpy as np
import cv2

class Shape(object):
    """
    Shape represents a shape, that is a figure created
    from a list of landmarks
    """

    def __init__(self, data):
        self._data = data

    @classmethod
    def from_points(cls, x_coord, y_coord):
        """
        from_points creates a Shape given two list of points

        Param:
            x_coord - x-coordinates of the Shape
            y_coord - y-coordinates of the Shape
        """

        if len(x_coord) != len(y_coord):
            raise AssertionError("Lenght should be equal")
        points = []
        for i, _ in enumerate(x_coord):
            points.append([x_coord[i], y_coord[i]])
        return cls(np.array(points))

    def mean(self):
        """
        Returns the mean of a Shape
        """
        return np.mean(self._data, axis=0)

    def _norm(self):
        """
        returns the norm of the data
        """
        return np.linalg.norm(self._data, axis=0)

    def shape(self):
        """
        returns the shape of the data
        """
        return self._data.shape

    def center(self):
        """
        centers the shape to the origin
        """
        return Shape(self._data - self.mean())

    def normalize(self):
        """
        normalize the data of a Shape
        """
        return Shape(self._data / self._norm())

    def align_parameters(self, other):
        """
        Align two parameters and returns the translation, scale factor and angle of rotation
        """

        this = self.collapse()
        other = other.collapse()

        this_length = len(this)/2
        other_length = len(other)/2

        # make sure both shapes are mean centered for computing scale and rotation
        this_centroid = np.array([np.mean(this[:this_length]), np.mean(this[this_length:])])
        other_centroid = np.array([np.mean(other[:other_length]), np.mean(other[other_length:])])
        this = [x - this_centroid[0] for x in this[:this_length]] + [y - this_centroid[1] for y in this[this_length:]]
        other = [x - other_centroid[0] for x in other[:other_length]] + [y - other_centroid[1] for y in other[other_length:]]

        # a = (x1.x2)/|x1|^2
        norm_this_sq = (np.linalg.norm(this)**2)
        a = np.dot(this, other) / norm_this_sq

        # b = sum_1->other_length(this_i*y2_i - y1_i*other_i)/|this|^2
        b = (np.dot(this[:this_length], other[other_length:]) 
             - np.dot(this[this_length:], other[:other_length])) / norm_this_sq

        # s^2 = a^2 + b^2
        scale = np.sqrt(a**2 + b**2)

        # theta = arctan(b/a)
        theta = np.arctan(b/a)

        # the optimal translation is chosen to match their centroids
        translation = other_centroid - this_centroid

        return translation, scale, theta


    def align(self, other):
        """
        return a new shape aligned with other shape
        """
        # get params
        _, s, theta = self.align_parameters(other)

        # align the two shapes
        this_data = self.rotate(theta)
        this_data = self.scale(s)

        # project into tangent space by scaling x1 with 1/(x1.x2)
        r = np.dot(this_data.collapse(), other.collapse())
        return Shape(this_data.collapse()*(1.0/r))

    def rotate(self, angle):
        """
        Returns a new shape rotate a given angle
        """
        # create rotation matrix
        matrix = np.array([[np.cos(angle), np.sin(angle)],
                           [-np.sin(angle), np.cos(angle)]])

        # apply rotation on each landmark point
        points = np.zeros_like(self._data)
        centroid = self.mean()
        tmp_points = self._data - centroid
        for ind in range(len(tmp_points)):
            points[ind, :] = tmp_points[ind, :].dot(matrix)
        points = points + centroid

        return Shape(points)


    def scale(self, factor):
        """
        Returns a new shape scaled by a factor
        """
        return Shape(factor * self._data)

    def translate(self, displacement):
        """
        Returns a new shape translated a displacement amount
        """
        return Shape(self._data + displacement)

    def collapse(self):
        """
        Return the data as a list [x1, x2, ..., xn, y1, y2, ..., yn]
        """
        return np.hstack((self._data[:, 0], self._data[:, 1]))

    def data(self):
        """
        Return the data as a list of points
        [[x1, y1], [x2, y2] ... [xn, yn]]
        """
        return self._data

    @staticmethod
    def matrix(shapes):
        """
        Returns the shapes as a numpy matrix.
        :return: a Nxd*d numpy matrix containing the shapes
        """
        return np.array([shape.collapse() for shape in shapes])
