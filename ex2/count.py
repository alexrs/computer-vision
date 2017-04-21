'''
Cell counting.

Computer Vision assigment.
Author: Alejandro Rodriguez Salamanca. r06509814

'''

import cv2
import cv2.cv
import numpy as np
import matplotlib.pyplot as plt
import math

def detect(img):
    '''
    Do the detection.
    '''
    #create a gray scale version of the image, with as type an unsigned 8bit integer
    img_g = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8 )
    img_g[:,:] = img[:,:,0]

    #1. Do canny (determine the right parameters) on the gray scale image
    edges = cv2.Canny(img, 50, 100, L2gradient=True)
    
    #Show the results of canny
    canny_result = np.copy(img_g)
    canny_result[edges.astype(np.bool)] = 0
    cv2.imshow('img', canny_result)
    cv2.waitKey(0)

    #2. Do hough transform on the gray scale image
    HIGH = 0
    LOW = 0
    
    circles = cv2.HoughCircles(img_g, cv2.cv.CV_HOUGH_GRADIENT, 2.0, 50.0, 30, 50, 30, 40, 70)
    circles = circles[0,:,:]
    
    #Show hough transform result
    showCircles(img, circles)
    
    #3.a Get a feature vector (the average color) for each circle
    nbCircles = circles.shape[0]
    features = np.zeros((nbCircles, 3), dtype=np.int)
    for i in range(nbCircles):
        features[i,:] = getAverageColorInCircle(img , int(circles[i,0]), int(circles[i,1]), int(circles[i,2]))
    
    #3.b Show the image with the features (just to provide some help with selecting the parameters)
    showCircles(img, circles, [str(features[i,:]) for i in range(nbCircles)])

    #3.c Remove circles based on the features
    selectedCircles = np.zeros((nbCircles), np.bool)
    for i in range(nbCircles):
        if in_range(features[i]):
            selectedCircles[i] = 1
            
    circles = circles[selectedCircles]

    #Show final result
    showCircles(img, circles)    
    return circles

def in_circle(center_x, center_y, radius, x, y):
    '''
    returns true if the coordinates x and y are inside the circle with center (cx, cy) and radius
    '''
    square_dist = (center_x - x) ** 2 + (center_y - y) ** 2
    return square_dist <= radius ** 2


def in_range(feature):
    '''
    Returns true if a feature is inside the allowed values of RGB. The order of the vector is BGR
    '''
    MIN_B = 155
    MAX_B = 220
    MIN_G = 145
    MAX_G = 195
    MIN_R = 165
    MAX_R = 210
    
    if feature[0] < MIN_B or feature[0] > MAX_B:
        return False
    if feature[1] < MIN_G or feature[1] > MAX_G:
        return False
    if feature[2] < MIN_R or feature[2] > MAX_R:
        return False
    return True
    
def getAverageColorInCircle(img, cx, cy, radius):
    '''
    Get the average color of img inside the circle located at (cx,cy) with radius.
    '''
    maxy,maxx,channels = img.shape
    C = np.zeros((3,))

    startx = max(cx - radius, 0)
    starty = max(cy - radius, 0)
    endx = min(cx + radius, maxx)
    endy = min(cy + radius, maxy)

    for j in range(starty, endy):
        for i in range(startx, endx):
            if in_circle(cx, cy, radius, i, j):
                C = np.vstack((C, img[j, i]))
    return np.mean(C, axis=0)
    
    
    
def showCircles(img, circles, text=None):
    '''
    Show circles on an image.
    @param img:     numpy array
    @param circles: numpy array 
                    shape = (nb_circles, 3)
                    contains for each circle: center_x, center_y, radius
    @param text:    optional parameter, list of strings to be plotted in the circles
    '''
    #make a copy of img
    img = np.copy(img)
    #draw the circles
    nbCircles = circles.shape[0]
    for i in range(nbCircles):
        cv2.circle(img, (int(circles[i,0]), int(circles[i,1])), int(circles[i,2]),
                   cv2.cv.CV_RGB(255, 0, 0), 2, 8, 0)
    #draw text
    if text != None:
        for i in range(nbCircles):
            cv2.putText(img, text[i], (int(circles[i,0]), int(circles[i,1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, cv2.cv.CV_RGB(0, 0,255))
    #show the result
    cv2.imshow('img',img)
    cv2.waitKey(0)    


        
if __name__ == '__main__':
    #read an image
    img = cv2.imread('normal.jpg')
    
    #print the dimension of the image
    print img.shape
    
    #show the image
    cv2.imshow('img',img)
    cv2.waitKey(0)
    
    #do detection
    circles = detect(img)
    
    #print result
    print "We counted "+str(circles.shape[0])+ " cells."
    








