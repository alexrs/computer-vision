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
        for i in range(len(x_coord)):
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

    def align_params(self, this, other):
        """Computes the optimal parameters for the alignment of two shapes.

        We wish to translate, scale and rotate x1 by (t, s, theta) so as to minimise
        |t+s*A*x1 - x2|, where A performs a rotation of a shape x by theta.

        Based on:
            An introduction to Active Shape Models - Appenix D

        Args:
            x1, x2: Two shapes with each as format [x0,x1...xn,y0,y1...yn].

        Returns:
            The optimal parameters t, s and theta to align x1 with x2.

        """
        # work in vector format
        x1 = this.collapse()
        x2 = other.collapse()

        l1 = len(x1)/2
        l2 = len(x2)/2

        # make sure both shapes are mean centered for computing scale and rotation
        x1_centroid = np.array([np.mean(x1[:l1]), np.mean(x1[l1:])])
        x2_centroid = np.array([np.mean(x2[:l2]), np.mean(x2[l2:])])
        x1 = [x - x1_centroid[0] for x in x1[:l1]] + [y - x1_centroid[1] for y in x1[l1:]]
        x2 = [x - x2_centroid[0] for x in x2[:l2]] + [y - x2_centroid[1] for y in x2[l2:]]

        # a = (x1.x2)/|x1|^2
        norm_x1_sq = (np.linalg.norm(x1)**2)
        a = np.dot(x1, x2) / norm_x1_sq

        # b = sum_1->l2(x1_i*y2_i - y1_i*x2_i)/|x1|^2
        b = (np.dot(x1[:l1], x2[l2:]) - np.dot(x1[l1:], x2[:l2])) / norm_x1_sq

        # s^2 = a^2 + b^2
        s = np.sqrt(a**2 + b**2)

        # theta = arctan(b/a)
        theta = np.arctan(b/a)

        # the optimal translation is chosen to match their centroids
        t = x2_centroid - x1_centroid

        return t, s, theta


    def align(self, other):
        """Aligns two mean centered shapes.

        Scales and rotates x1 by (s, theta) so as to minimise |s*A*x1 - x2|,
        where A performs a rotation of a shape x by theta.

        Based on:
            An introduction to Active Shape Models - Appenices A & D

        Args:
            x1: The shape which will be scaled and rotated.
            x2: The shape to which x1 will be aligned.

        Returns:
            The aligned version of x1.

        """
        
        # get params
        this_data = self._data
        other_data = other.data()
        _, s, theta = self.align_params(this_data, other_data)

        # align the two shapes
        this_data = this_data.rotate(theta)
        this_data = this_data.scale(s)

        # project into tangent space by scaling x1 with 1/(x1.x2)
        r = np.dot(this_data.collapse(), other_data.collapse())
        return Shape(this_data.collapse()*(1.0/r))

    def scale(self, factor):
        return Shape(factor * self._data)

    def translate(self, displacement):
        return Shape(self._data + displacement)

    def collapse(self):
        n, _ = self._data.shape
        return np.reshape(self._data, 2 * n)

    def data(self):
        return self._data

    def rotation(self, theta, s):
        """
         Construct rotation and translation matrix (M and t)
        """
        M = np.array([[s*np.cos(theta), -s*np.sin(theta)], 
            [s*np.sin(theta),  s*np.cos(theta)]]).reshape(2, 2)
        return M

class AlignedShape(object):

    def __init__(self, shapes, tol=1e-7, max_iters=10000):
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

