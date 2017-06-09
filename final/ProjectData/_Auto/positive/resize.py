from os import listdir
from os.path import isfile, join
import cv2

onlyfiles = [f for f in listdir(".") if isfile(join(".", f))]


for f in onlyfiles:
    if "jpg" in f:
        print "Modifying file: ", f
        img = cv2.imread(f, 0)
#        img1 = img[280:560, 0:420]
        img = img[:560,:420]

        img = cv2.resize(img, (75, 100))
#        img1 = cv2.resize(img1, (100,100))
#        cv2.imshow("i", img)
#        cv2.imshow("ii", img1)

#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
        cv2.imwrite(f, img)
#        cv2.imwrite("m_"+f, img1)
