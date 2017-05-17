"""
Authors: Alejandro Rodriguez, Fernando Collado
"""
import numpy as np
import plot

class PCA:
    """
    """

    def __init__(self, data, dims=13):
        self._data = data
        self._eigs = []
        self._eves = []
        self._mean = 0
        self._pca(dims)

    def _pca(self, dims_rescaled_data):
        """
        returns: data transformed in 2 dims/columns + regenerated original data
        pass in: data as 2D NumPy array
        """
        m, n = self._data.shape
        # mean center the data
        self._mean = np.mean(self._data, axis=0)
        # calculate the covariance matrix
        cov = np.cov(self._data.T)
        # calculate eigenvectors & eigenvalues of the covariance matrix
        self._evals, self._evecs = np.linalg.eigh(cov)
        #plot.variance(self._evals)
        # sort eigenvalue in decreasing order
        idx = np.argsort(self._evals)[::-1]
        self._evecs = self._evecs[:,idx]
        # sort eigenvectors according to same index
        self._evals = self._evals[idx]
        # select the first n eigenvectors (n is desired dimension
        # of rescaled data array, or dims_rescaled_data)
        self._evecs = self._evecs[:, :dims_rescaled_data]

    def project(self):
        """
        """
        return np.dot(self._evecs.T, (self._data - self._mean).T)

    def reconstruct(self):
        """
        """
        return np.dot(self._evecs, self.project()).T + self._mean

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

    def test_PCA(self, data):
        """
        test by attempting to recover original data array from
        the eigenvectors of its covariance matrix & comparing that
        'recovered' array with the original data
        """
        data_recovered = self.reconstruct()
        print np.allclose(data, data_recovered)
