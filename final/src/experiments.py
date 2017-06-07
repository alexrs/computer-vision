"""
Authors: Alejandro Rodriguez, Fernando Collado
"""
from dataset import Dataset
from enhacement import Enhancement
from utils import Utils
from plot import Plot
import cv2

def save(name, img, path="../ProjectData/_Data/Radiographs/Experiments"):
    Utils.create_dir(path)
    cv2.imwrite("{}/{}.tif".format(path, name), img)

if __name__ == '__main__':

    
    # dataset = Dataset()
    # X = []
    # for i in range(1, 9):
    #     X.append(dataset.load(i)[0])

    # imgs = dataset.get_images()
    # test_img = imgs[0]

    # Plot.approximated_shape(X, test_img, wait=True)



    # img = dataset.get_images("Experiments/")[0]
    # img = Enhancement.fastNlMeansDenoising(img)
    # save("denoised", img)
    # top_hat_img = Enhancement.top_hat(img)
    # black_hat_img = Enhancement.black_hat(img)
    # img = img + top_hat_img
    # save("top_hat", img)
    # img = img - black_hat_img
    # save("black_hat", img)
    # img = Enhancement.clahe(img)
    # save("clahe", img)
    # img = Enhancement.sobel(img)
    # save("sobel", img)
