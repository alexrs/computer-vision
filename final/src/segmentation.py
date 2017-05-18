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

INCISOR = 1 # The incisor we want to segmentate
RADIOGRAPH = 1 # The radiograph we want to use

def main():
    """
    Main function of the incisor segmentation project
    """

    # leave-one-out
    train_indices = range(0, 14)
    train_indices.remove(RADIOGRAPH-1)

    # Get the dataset
    dataset = Dataset()
    # Load landmarks
    landmarks = dataset.load_mirrored(INCISOR)

    test_data = landmarks[RADIOGRAPH-1]
    train_data = [landmarks[index] for index in train_indices]

    imgs = dataset.get_images()
    test_img = imgs[RADIOGRAPH-1]
    train_imgs = [imgs[index] for index in train_indices]

    # train
    model = ActiveShapeModel(train_data)
    print model.mean_shape().data()

    # preprocess the landmarks
    Plot.landmarks(model.mean_shape().data())
    # Load radiographs
    img = Enhancement.sobel(imgs[RADIOGRAPH - 1])
    #plot.image(img)

    Init(model.mean_shape(), imgs[RADIOGRAPH - 1])
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
