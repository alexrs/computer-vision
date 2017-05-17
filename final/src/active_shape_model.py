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
        # Perform procrustes analysis
        self._procrustes = Procrustes(shapes)
        # get the aligned shapes
        self._aligned_shapes = Shape.matrix(self._procrustes.get_aligned_shapes())
        # perform PCA on the aligned shapes
        self._pca = PCA(self._aligned_shapes)

    def pca(self):
        """
        Returns the PCA of the model
        """
        return self._pca

    def mean_shape(self):
        """
        Returns the mean shape of the model
        """
        return self._procrustes.get_mean_shape()


