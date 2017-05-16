"""
Authors: Alejandro Rodriguez, Fernando Collado
"""
import numpy as np
import cv2
import cv2.cv as cv
from dataset import Dataset
import plot
from active_shape_model import ActiveShapeModel
from enhacement import Enhancement

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
    #plot.landmarks(model.mean_shape().data())
    #pca.test_PCA(aligned.data())
    # Load radiographs
    imgs = dataset.get_images()
    img = Enhancement.sobel(imgs[0])
    plot.image(img)
    # Improve quality of dental radiographs

    #img = cv2.fastNlMeansDenoising(imgs[0], 10, 10, 7, 21)
    # equalize the histogram of the input image
    #histeq = cv2.equalizeHist(img)
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
