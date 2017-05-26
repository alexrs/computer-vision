"""
Authors: Alejandro Rodriguez, Fernando Collado
"""
import numpy as np
from plot import Plot

class PCA:
    """
    """

    def __init__(self, data, dims=4):
        self._data = data
        self._eigs = []
        self._eves = []
        self._pc_modes = []
        self._mean = 0
        self._pca(dims)

    def _pca(self, dims):
        """
        performs PCA on a given data.

        saves the modes of variation, as stated in the paper of Cootes
        """
        # mean center the data
        self._mean = np.mean(self._data, axis=0)
        # calculate the covariance matrix
        cov = np.cov(self._data, rowvar=0)
        # calculate eigenvectors and eigenvalues of the covariance matrix
        self._evals, self._evecs = np.linalg.eigh(cov)
        # Plot.variance(self._evals)
        # sort eigenvalue in decreasing order
        idx = np.argsort(-self._evals)
        self._evecs = self._evecs[:, idx]
        # sort eigenvectors according to same index
        self._evals = self._evals[idx]

        matrix = []
        for i in range(dims):
            matrix.append(np.sqrt(self._evals[i]) * self._evecs[:, i])

        self._pc_modes = np.array(matrix).T

    def pc_modes(self):
        """
        returns the modes of variation
        """
        return self._pc_modes

    def mean(self):
        """
        return the mean
        """
        return self._mean

    def eigenvalues(self):
        """
        return the eigenvalues
        """
        return self._evals

    def eigenvectors(self):
        """
        return the eigenvectors
        """
        return self._evecs
