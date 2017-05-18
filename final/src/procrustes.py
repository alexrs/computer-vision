"""
Authors: Alejandro Rodriguez, Fernando Collado
"""

import numpy as np
from shape import Shape

class Procrustes(object):
    """
    This class encapsulates the classes and methods needed to perform
    Procruestes Analysis

    See: Appendix A: Aligning a pair of shapes, from 'Active Shape Models - Their Training
            and Application', Cootes et al.
         Appendix A: 'An Introduction to Active Shape Models', Cootes.
         Slide 8 of http://www.robots.ox.ac.uk/~jmb/lectures/InformaticsLecture6.pdf
         http://www.cse.psu.edu/~rtc12/CSE586/lectures/cse586Shape1.pdf
    """

    def __init__(self, shapes, max_iters=1000, tol=1e-10):
        self._procrustes(shapes, max_iters, tol)

    def _procrustes(self, shapes, max_iters, tol):
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
        # translate each example so that its centre of gravity is at the origin.
        self._aligned_shapes = [shape.center() for shape in shapes]

        # 2: choose one example as an initial estimate of the mean shape
        # and scale so that it is normalized
        self._mean_shape = self._aligned_shapes[0].normalize()
        # iterate
        for i in range(max_iters):
            # 4: align all shapes with current estimate of mean shape
            for j, shape in enumerate(self._aligned_shapes):
                self._aligned_shapes[j] = self.align(shape, self._mean_shape)

            # 5: re-estimate the mean from aligned shapes
            new_mean_shape = self._new_mean_shape(self._aligned_shapes)

            # 6: apply constraints on scale and orientation to the current estimate
            # of the mean by aligning it with x0 and scaling so that |x| = 1.
            new_mean_shape = self.align(new_mean_shape, self._mean_shape)
            new_mean_shape = new_mean_shape.normalize().center()

            # 7: if converged, do not return to 4
            if ((self._mean_shape.collapse() - new_mean_shape.collapse()) < tol).all():
                print "Procrustes", i
                #print self._mean_shape.data()
                break

            self._mean_shape = new_mean_shape


    def align(self, x1, x2):
        """
        """

        # get params
        _, s, theta = self.align_params(x1, x2)

        # align x1 with x2
        x1 = x1.rotate(theta)
        x1 = x1.scale(s)

        # project into tangent space by scaling x1 with 1/(x1.x2)
        xx = np.dot(x1.collapse(), x2.collapse())
        return Shape.from_list(x1.collapse()*(1.0/xx))


    def align_params(self, x1, x2):
        """
        """
        # work in vector format
        x1 = x1.collapse()
        x2 = x2.collapse()

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

    def get_mean_shape(self):
        """
        returns the mean shape
        """
        return self._mean_shape

    def get_aligned_shapes(self):
        """
        returns the aligned shapes
        """
        return self._aligned_shapes

    def _new_mean_shape(self, aligned_shapes):
        """
        """
        mean_shape = []
        for shape in aligned_shapes:
            mean_shape.append(shape.collapse())
        mean_shape = np.array(mean_shape)
        return Shape.from_list(np.mean(mean_shape, axis=0))

