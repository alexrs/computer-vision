import numpy as np
import cv2
import cv2.cv as cv
from pca import pca, test_PCA
from dataset import Dataset
import plot
from shape import Shape, AlignedShape

if __name__ == "__main__":
    dataset = Dataset("../ProjectData/_Data/")
    # Load landmarks
    landmarks = dataset.get_landmarks()
    # preprocess the landmarks
    landmarks = landmarks[::8] # try first with one tooth
    aligned = AlignedShape(landmarks)
    plot.plot_landmarks(aligned.mean_shape().data)
    # PCA
    # dat, eigenvalues, eigenvectors, mu = pca(mtx2)
    # test_PCA(mtx2)
    # Load radiographs
    imgs = dataset.get_images()
    #filenames = fnmatch.filter(os.listdir(directory),'*.tif')
    #for fname in filenames:
        #img = cv2.imread(directory + "/" + fname, 0)
        # Improve quality of dental radiographs
       # img = cv2.fastNlMeansDenoising(img, 10, 10, 7, 21)
        # equalize the histogram of the input image
       # histeq = cv2.equalizeHist(img)
       # cv2.imshow("img",histeq)
       # cv2.waitKey(0)
        
        
