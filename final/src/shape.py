"""
Authors: Alejandro Rodriguez, Fernando Collado
"""
import numpy as np

class Shape(object):
    """
    Shape represents a shape, that is a figure created
    from a list of landmarks
    """

    def __init__(self, data):
        self._data = data

    @staticmethod
    def matrix(shapes):
        """
        Returns the shapes as a numpy matrix.
        :return: a Nxd*d numpy matrix containing the shapes
        """
        return np.array([shape.collapse() for shape in shapes])

    @classmethod
    def from_list_file(cls, landmark):
        """
        from_list creates a Shape given a list of points [x1, y1, x2, y2, ... , xn, yn]
        """
        x = []
        y = []
        for i, line in enumerate(landmark):
            if i % 2 == 0:
                x.append(float(line))
            else:
                y.append(float(line))
        return Shape.from_points(x, y)

    @classmethod
    def from_list_consecutive(cls, landmark):
        """
        Creates a Shape from a list of points in format [x1, x2, ..., y1, y2, ...]
        This arrangement is used in the papers
        "Modified Grey-level Models for Active Shape Model Training"
        """
        mid = len(landmark)/2
        return Shape(np.array((landmark[:mid], landmark[mid:])).T)

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

    def transform(self, translate, scale, theta):
        """
        Performs a transformation, rotating by theta, scaling by scale, and translating by traslate
        """
        return self.rotate(theta).scale(scale).translate(translate)

    def inverse_transform(self, translate, scale, theta):
        """
        Performs the inverse transformation of the given arguments
        """
        return self.translate(-translate).scale(1/scale).rotate(-theta)


    def mean(self):
        """
        Returns the mean of a Shape
        """
        return np.mean(self._data, axis=0)

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
        http://kawahara.ca/how-to-normalize-vectors-to-unit-norm-in-python/
        """
        total = np.sqrt(np.power(self.center().data(), 2).sum())
        data = self._data.dot(1. / total)
        return Shape(data)

    def rotate(self, angle):
        """
        Returns a new shape rotate a given angle
        """
        # create rotation matrix
        matrix = np.array([[np.cos(angle), np.sin(angle)],
                           [-np.sin(angle), np.cos(angle)]])

        # apply rotation
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
        mean = self.mean()
        points = (self._data - mean).dot(factor) + mean
        return Shape(points)

    def translate(self, displacement):
        """
        Returns a new shape translated a displacement amount
        """
        return Shape(self._data + displacement)

    def collapse(self):
        """
        Return the data as a list [x1, y1, ..., xn, yn]
        """
        return np.hstack((self._data[:, 0], self._data[:, 1]))

    def data(self):
        """
        Return the data as a list of points
        [[x1, y1], [x2, y2] ... [xn, yn]]
        """
        return self._data

    def mirror_y(self):
        """
        Mirrors this model around the y axis.
        """
        mean = self.mean()
        points = self._data - mean
        points[:, 0] *= -1
        points = points + mean
        points = points[::-1]
        return Shape(points)
