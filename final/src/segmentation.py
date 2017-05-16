"""
Authors: Alejandro Rodriguez, Fernando Collado
"""
import numpy as np
import cv2
import cv2.cv as cv
from pca import PCA
from dataset import Dataset
import plot
from shape import Shape, AlignedShape

INCISOR = 1 # The incisor we want to segmentate
RADIOGRAPH = 1 # The radiograph we want to use

def main():
    """
    Main function of the incisor segmentation project
    """
    # Get the dataset
    dataset = Dataset()
    # Load landmarks
    landmarks = dataset.get_landmarks(INCISOR)
    # preprocess the landmarks
    model = ActiveShapeModel(landmarks)
    
    aligned = AlignedShape(landmarks)
    #plot.landmarks(aligned.data())
    # PCA
    pca = PCA(aligned.data())
    #pca.test_PCA(aligned.data())
    # Load radiographs
    imgs = dataset.get_images()
    # Improve quality of dental radiographs
    img = cv2.fastNlMeansDenoising(imgs[0], 10, 10, 7, 21)
    # equalize the histogram of the input image
    histeq = cv2.equalizeHist(img)
    #plot.image(histeq)
    #cv2.setMouseCallback("img", set_point(img))
    #cv2.waitKey(0)


def find(incisor, radiograph):
    """
    find finds an incisor in a given radiograph
    """
    pass

if __name__ == "__main__":
    main()
