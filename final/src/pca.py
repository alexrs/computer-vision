import numpy as np
import plot

def pca(data, dims_rescaled_data=3):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    m, n = data.shape
    # mean center the data
    mu = np.mean(data, axis=0)
    X = data
    # calculate the covariance matrix
    cov = np.cov(X)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric, 
    # the performance gain is substantial
    evals, evecs = np.linalg.eig(cov)
    plot.plot_variance(evals)    
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, eigenvectors and mean
    return np.dot(evecs.T, (X - mu)), evals, evecs, mu

def test_PCA(data):
    '''
    test by attempting to recover original data array from
    the eigenvectors of its covariance matrix & comparing that
    'recovered' array with the original data
    '''
    m , _ , eigenvectors, mu = pca(data)
    data_recovered = np.dot(eigenvectors, m) + mu
    print np.allclose(data, data_recovered)
