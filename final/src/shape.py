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

    def align_params(self, other):
        """
        TODO: Comment
        """

        this = self.collapse()
        other = other.collapse()

        l1 = len(this)/2
        l2 = len(other)/2

        # make sure both shapes are mean centered for computing scale and rotation
        this_centroid = np.array([np.mean(this[:l1]), np.mean(this[l1:])])
        other_centroid = np.array([np.mean(other[:l2]), np.mean(other[l2:])])
        this = [x - this_centroid[0] for x in this[:l1]] + [y - this_centroid[1] for y in this[l1:]]
        other = [x - other_centroid[0] for x in other[:l2]] + [y - other_centroid[1] for y in other[l2:]]

        # a = (x1.x2)/|x1|^2
        norm_this_sq = (np.linalg.norm(this)**2)
        a = np.dot(this, other) / norm_this_sq

        # b = sum_1->l2(this_i*y2_i - y1_i*other_i)/|this|^2
        b = (np.dot(this[:l1], other[l2:]) - np.dot(this[l1:], other[:l2])) / norm_this_sq

        # s^2 = a^2 + b^2
        s = np.sqrt(a**2 + b**2)

        # theta = arctan(b/a)
        theta = np.arctan(b/a)

        # the optimal translation is chosen to match their centroids
        t = other_centroid - this_centroid

        return t, s, theta


    def align(self, other):
        """
        TODO: Comment
        """
        # get params
        _, s, theta = self.align_params(other)

        # align the two shapes
        this_data = self.rotate(theta)
        this_data = self.scale(s)

        # project into tangent space by scaling x1 with 1/(x1.x2)
        r = np.dot(this_data.collapse(), other.collapse())
        return Shape(this_data.data() / r)

    def rotate(self, angle):
        """
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
        TODO: comment
        """
        return Shape(factor * self._data)

    def translate(self, displacement):
        """
        TODO: Comment
        """
        return Shape(self._data + displacement)

    def collapse(self):
        """
        TODO: Comment
        """
        n, _ = self._data.shape
        return np.reshape(self._data, 2 * n)

    def data(self):
        """
        TODO: Comment
        """
        return self._data

    @staticmethod
    def matrix(shapes):
        """
        Returns the shapes as a numpy matrix.
        :return: a Nxd*d numpy matrix containing the shapes
        """
        return np.array([shape.collapse() for shape in shapes])
