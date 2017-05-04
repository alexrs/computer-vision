import numpy as np
import cv2
import cv2.cv as cv
import os
import fnmatch
import matplotlib.pyplot as plt


def load_landmarks(directory, dim=80):
    '''
    load_landmarks creates an array that contains all the landmarks of a
    directory.
    @return np.array, shape=(num of files, num of landmarks in each file=80)
    '''
    #get the filenames that matches the extension
    filenames = fnmatch.filter(os.listdir(directory), '*.txt')
    # create the array
    X = np.zeros((len(filenames), dim))

    # iterate over the files to create a matrix containing the landmarks
    for i, filename in enumerate(filenames):
        file_in = directory + "/" + filename
        lands = np.loadtxt(file_in, dtype=float)
        X[i, :] = lands

    return X

def procrustes(X, Y, scaling=True, reflection='best'):
    """
    https://stackoverflow.com/questions/18925181/procrustes-analysis-with-numpy
    
    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y    
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling 
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d       
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform   
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        zeros =  np.zeros((n, m-my))
        Y0 = np.hstack((Y0, zeros))

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:
        # optimum scaling of Y
        b = traceTA * normX / normY
        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2
        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b * np.dot(muY, T)

    #transformation values 
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform

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
    #plot_variance(evals)    
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return np.dot(evecs.T, (X - mu)), evals, evecs, mu

def plot_variance(eig_vals):
    tot = sum(eig_vals)
    var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(6, 4))
        plt.bar(range(80), var_exp, alpha=0.5, align='center',
                label='individual explained variance')
        plt.step(range(80), cum_var_exp, where='mid',
                 label='cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()


def plot_landmarks(X):
    x = []
    y = []
    for i, Y in enumerate(X):
        for j, elem in enumerate(Y):
            if j % 2 == 0:
                x.append(elem)
            else:
                y.append(elem)

    plt.plot(x, y, '.')
    plt.show()

def test_PCA(data):
    '''
    test by attempting to recover original data array from
    the eigenvectors of its covariance matrix & comparing that
    'recovered' array with the original data
    '''
    m , _ , eigenvectors, mu = pca(data)
    data_recovered = np.dot(eigenvectors, m) + mu
    print data
    print
    print
    print
    print data_recovered
    print np.allclose(data, data_recovered)

if __name__ == "__main__":
    # Load landmarks
    landmarks = load_landmarks("ProjectData/_Data/Landmarks/original")
    # preprocess the landmarks 
    d, Z, tform = procrustes(landmarks.T, landmarks[0:1].T)
    # PCA
    dat, eigenvalues, eigenvectors, mu = pca(Z)
    #plot_landmarks(dat)
    # Load radiographs
    directory = "ProjectData/_Data/Radiographs"
    filenames = fnmatch.filter(os.listdir(directory),'*.tif')
    for fname in filenames:
        img = cv2.imread(directory + "/" + fname, 0)
        # Improve quality of dental radiographs
       # img = cv2.fastNlMeansDenoising(img, 10, 10, 7, 21)
        # equalize the histogram of the input image
       # histeq = cv2.equalizeHist(img)
       # cv2.imshow("img",histeq)
       # cv2.waitKey(0)
        
        
