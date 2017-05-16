""""
Authors: Alejandro Rodriguez, Fernando Collado
"""

from procrustes import Procrustes
from pca import PCA
from shape import Shape

class ActiveShapeModel(object):
    """
    This class encapsulates the logic to compute the Active Shape Model of a set of Shapes
    """
    def __init__(self, shapes):
        self._procrustes = Procrustes(shapes)

        self._mean_shape = self._procrustes.get_mean_shape()
        self._aligned_shapes = Shape.matrix(self._procrustes.get_aligned_shapes())

        self._pca = PCA(self._aligned_shapes)

    
    def pca(self):
        """
        """
        return self._pca

    def mean_shape(self):
        """
        """
        return self._mean_shape


