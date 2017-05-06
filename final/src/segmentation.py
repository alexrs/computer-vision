import numpy as np
import cv2
import cv2.cv as cv
from scipy import spatial
from pca import pca
import load
import plot
from procrustes import procrustes

if __name__ == "__main__":
    # Load landmarks
    landmarks = load.load_landmarks("../ProjectData/_Data/Landmarks/original")
    # preprocess the landmarks
    #landmarks = landmarks[::8]
    mtx1, mtx2, disparity = procrustes(landmarks)
    # PCA
    dat, eigenvalues, eigenvectors, mu = pca(mtx2)
    test_PCA(mtx2)
    # plot_landmarks(dat)
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
        
        
