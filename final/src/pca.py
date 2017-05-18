"""
Authors: Alejandro Rodriguez, Fernando Collado
"""
import numpy as np
import plot

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
        returns: data transformed in 2 dims/columns + regenerated original data
        pass in: data as 2D NumPy array
        """
        # mean center the data
        self._mean = np.mean(self._data, axis=0)
        # calculate the covariance matrix
        cov = np.cov(self._data, rowvar=0)
        # calculate eigenvectors & eigenvalues of the covariance matrix
        self._evals, self._evecs = np.linalg.eigh(cov)
        #plot.variance(self._evals)
        # sort eigenvalue in decreasing order
        idx = np.argsort(-self._evals)
        self._evecs = self._evecs[:, idx]
        # sort eigenvectors according to same index
        self._evals = self._evals[idx]
        # select the first n eigenvectors (n is desired dimension
        # of rescaled data array, or dims)
        M = []
        for i in range(0, dims - 1):
            M.append(np.sqrt(self._evals[i]) * self._evecs[:, i])
        self.pc_modes = np.array(M).squeeze().T

    def get_pc_modes(self):
        """
        """
        return self._pc_modes

    def mean(self):
        """
        """
        return self._mean

    def eigenvalues(self):
        """
        """
        return self._evals

    def eigenvectors(self):
        """
        """
        return self._evecs
