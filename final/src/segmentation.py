"""
Authors: Alejandro Rodriguez, Fernando Collado

https://upcommons.upc.edu/bitstream/handle/2117/88341/gonzalo.lopez_103365.pdf?sequence=1&isAllowed=y
http://www.face-rec.org/algorithms/AAM/app_models.pdf
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.17.1120&rep=rep1&type=pdfs
https://www.reddit.com/r/opencv/comments/28hjwb/how_to_find_initial_position_for_an_active_shape/
"""
from dataset import Dataset
from plot import Plot
from active_shape_model import ActiveShapeModel
from enhacement import Enhancement
from init import Init
from fit import Fit
from grey_level_model import GreyLevelModel

import cv2

INCISOR = 1  # The incisor we want to segmentate
RADIOGRAPH = 1  # The radiograph we want to use


def main():
    """
    Main function of the incisor segmentation project
    """

    # leave-one-out
    train_indices = range(14) # list from 0 to 13 that coincides with the number of images
    train_indices.remove(RADIOGRAPH - 1)

    # Get the dataset
    dataset = Dataset()
    # Load landmarks
    landmarks = dataset.load_mirrored(INCISOR)
    
    # Divide between test data and train data
    test_data = landmarks[RADIOGRAPH - 1]
    train_data = [landmarks[index] for index in train_indices]

    # Get images
    imgs = dataset.get_images()
    # Divide between test images and train images
    test_img = imgs[RADIOGRAPH - 1]
    train_imgs = [imgs[index] for index in train_indices]

    # Create the Active Shape Model
    asm = ActiveShapeModel(train_data)

    # Create the Grey Level Model
    glm = GreyLevelModel()
    grey_modes = glm.get_modes()

    # Get the initial position
    init = Init(asm.mean_shape(), imgs[RADIOGRAPH - 1])
    initial_fit = init.get_initial_fit()

    # Fit the model to the image
    fit = Fit(initial_fit, test_img)

    # Evaluation of the results
    # TODO

if __name__ == "__main__":
    main()
