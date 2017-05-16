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
    """

    def __init__(self, shapes, max_iters=10000, tol=1e-7):
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
        # Align shapes and select a 'base' shape to be the reference in aligment
        self._aligned_shapes = [shape.center() for shape in shapes]
        self._mean_shape = self._aligned_shapes[0].normalize()

        # iterate until max_iters is reached or the variation is lower than tol
        for _ in range(max_iters):
            # align all the shapes
            for i, shape in enumerate(self._aligned_shapes):
                self._aligned_shapes[i] = shape.align(self._mean_shape)

            # select a new mean_shape
            previous_mean_shape = self._mean_shape
            self._mean_shape = self._new_mean_shape(self._aligned_shapes)

            # Check if the value of the mean shape has varied less than the tolerance
            if np.linalg.norm(self._mean_shape.data() - previous_mean_shape.data()) < tol:
                break


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
        return Shape(np.mean(np.array([shape.data() for shape in aligned_shapes]),
                             axis=0)).center().normalize()

