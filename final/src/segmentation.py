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
from fitter import Fitter
from grey_level_model import GreyLevelModel
import cv2

INCISOR = 3  # The incisor we want to segmentate
RADIOGRAPH = 1  # The radiograph we want to use
NUM_LANDMARKS = 40 # Number of points in a file

def main():
    """
    Main function of the incisor segmentation project
    """

    print "Loading data..."
    # leave-one-out
    train_indices = range(14) # list from 0 to 13 that coincides with the number of images
    train_indices.remove(RADIOGRAPH - 1)

    # Get the dataset
    dataset = Dataset()

    # Load landmarks
    shapes = dataset.load_mirrored(INCISOR)

    # Divide between test data and train data
    test_data = shapes[RADIOGRAPH - 1]
    train_data = [shapes[i] for i in train_indices]

    # Get images
    imgs = dataset.get_images()

    # Divide between test images and train images
    test_img = imgs[RADIOGRAPH - 1]
    train_imgs = [imgs[i] for i in train_indices]

    print "Creating Active Shape Model..."
    # Create the Active Shape Model
    asm = ActiveShapeModel(train_data)
    #Plot.active_shape_model(asm)

    print "Enhancing images..."
    # check if the enhanced images are stored
    if dataset.is_enhanced(): # if they are already stored, load from disk
        enhanced_imgs = dataset.get_enhanced()
    else: # if not, get the enhanced images (this will take a while)
        enhanced_imgs = [Enhancement.enhance(img, i, save=True) for i, img in enumerate(imgs)]

    test_enhanced_img = enhanced_imgs[RADIOGRAPH - 1]
    train_enhanced_imgs = [enhanced_imgs[i] for i in train_indices]

    # Create the Grey Level Model
    print "Creating Grey Level Models..."
    gl_models = []
    for i in range(NUM_LANDMARKS):
        grey_mode = GreyLevelModel(train_imgs, enhanced_imgs, train_data, i)
        gl_models.append(grey_mode)

    # Get the initial position
    print "Computing initial fit"
    init = Init(asm.mean_shape(), imgs[RADIOGRAPH - 1])
    initial_fit = init.get_initial_fit()

    # Fit the model to the image
    print "Fitting the model..."
    fitter = Fitter(initial_fit, test_img, test_enhanced_img,
                    gl_models, asm.pca().pc_modes(), test_data)
    X = fitter.fit()


    # Evaluation of the results
    # TODO
    Plot.approximated_shape([test_data, X], test_img, wait=True)

if __name__ == "__main__":
    main()
