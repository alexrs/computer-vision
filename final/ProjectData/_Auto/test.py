import numpy as np
import cv2

teeth_cascade = cv2.CascadeClassifier('cascade.xml')

img = cv2.imread('../_Data/Radiographs/01.tif', 0)

print img.shape
img_t = img[600:1400, 1000:2000]

print img_t.shape


tooth = teeth_cascade.detectMultiScale(img_t)

print tooth

for (x,y,w,h) in tooth:
    cv2.rectangle(img_t,(x,y),(x+w,y+h),(255,0,0),2)

cv2.imshow('img',img_t)
cv2.waitKey(0)
cv2.destroyAllWindows()
