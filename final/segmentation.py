import numpy as np
import cv2
import cv2.cv as cv
import os

def load_landmarks(directory, dim=80):
    '''
    load_landmarks creates an array that contains all the landmarks of a
    directory.
    @return np.array, shape=(num of files, num of landmarks in each file=80)
    '''
    #get the filenames that matches the extension
    filenames = fnmatch.fiter(os.listdir(directory), '*.txt')
    # create the array
    X = np.zeros((len(filenames), dim))

    # iterate over the files to create a matrix containing the landmarks
    for i, filename in enumerate(filenames):
        file_in = directory + "/" + filename
        lands = np.loadtxt(file_in)
        X[i, :] = lands

    return X

if __name__ == "__main__":
    # Load landmarks
    X = load_landmarks("ProjectData/_Data/Landmarks/original")
    print X
    
