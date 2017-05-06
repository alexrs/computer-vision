import numpy as np
from scipy import spatial

def procrustes(X):
    '''
    https://stackoverflow.com/questions/18925181/procrustes-analysis-with-numpy
    '''
    m, n = X.shape
    zeros =  np.zeros((n, m - 1))
    Y0 = np.hstack((X[0:1].T, zeros))

    return spatial.procrustes(X.T, Y0)
