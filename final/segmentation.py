import numpy as np
import cv2
import cv2.cv as cv
import os
import fnmatch

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
        lands = np.loadtxt(file_in)
        X[i, :] = lands

    return X

def preprocess_landmarks(points1, points2):
    '''
    preprocess_landmarks preprocess the landmarks to normalise translation, rotation
    and scale differences (Procrustes Analysis)

    @return np.array with preprocessed data
    '''

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])

def pca(X):
    '''
    http://sebastianraschka.com/Articles/2015_pca_in_3_steps.html#pca-vs-lda
    ''' 
    mean = np.mean(X, axis=0)
    cov = np.cov(X.T)
    eig_vals, eig_vecs = np.linalg.eig(cov)
    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    tot = sum(eig_vals)
    
    var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    plot_variance(var_exp, cum_var_exp)

    # TODO - CHANGE THIS
    matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),
                          eig_pairs[1][1].reshape(4,1)))
    Y = X_std.dot(matrix_w)
    return Y

def plot_variance(var_exp, cum_var_exp):
    with plt.style.context('seaborn-whitegrid'):

        plt.figure(figsize=(6, 4))
        plt.bar(range(4), var_exp, alpha=0.5, align='center',
                label='individual explained variance')
        plt.step(range(4), cum_var_exp, where='mid',
                 label='cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.legend(loc='best')
        plt.tight_layout()



if __name__ == "__main__":
    # Load landmarks
    X = load_landmarks("ProjectData/_Data/Landmarks/original")
    # preprocess the landmarks 
    X = preprocess_landmarks(X)
    # PCA
    red_X = pca(X)
    # Load radiographs
    
    # Improve quality of dental radiographs
    
    
    
    
    print X
    
