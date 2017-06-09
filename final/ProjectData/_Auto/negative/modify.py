from os import listdir
from os.path import isfile, join
import cv2

onlyfiles = [f for f in listdir(".") if isfile(join(".", f))]


for f in onlyfiles:
    #if "jpg" in f and len(f)<7:
    if "jpg" in f :

        print "Modifying file: ", f
        img = cv2.imread(f, 0)
#        img = img[:560,:424]
#        img = cv2.resize(img, (75, 100))
        cv2.imwrite(f, img)
