import cv2
import numpy as np
from matplotlib import pyplot as plt
from dataset import Dataset
import xml.etree.ElementTree as ET
import re
import pickle


ppath = "../ProjectData/_Auto/positive/"
npath = "../ProjectData/_Auto/negative/"
txtPath = "../ProjectData/_Data/Landmarks/original/landmarks"

#img = cv2.imread(ppath+"01.tif",0)
#cv2.imshow("img", img)
#cv2.waitKey(0)

#plt.hist(img.ravel(),256,[0,256])
#plt.show()


#img = np.concatenate((img, np.array(img[:,437]).reshape(336,1)), axis=1)


training_set = []
training_labels = []

svm = cv2.SVM()

try:
    tree = ET.parse('svm.xml')
except:
    for i in range(1, 31):
        if i<10:
            i_path = ppath + "0"+str(i)+".tif"
        else:
            i_path = ppath  + str(i)+".tif"
     
        img = cv2.imread(i_path,0)
        img = img[:560, :424]   
        xarr=np.squeeze(np.array(img).astype(np.float32))
        arr= np.array(xarr)
        flat_arr= arr.ravel()
        training_set.append(flat_arr)
        training_labels.append(1)

    for i in range(1,15):
        i_path = npath + str(i)+".jpg"
     
        img = cv2.imread(i_path,0)
        img = img[:560,:424]   
        xarr=np.squeeze(np.array(img).astype(np.float32))
        arr= np.array(xarr)
        flat_arr= arr.ravel()
        training_set.append(flat_arr)
        training_labels.append(0)


    svm_params = dict( kernel_type = cv2.SVM_LINEAR, 
                           svm_type = cv2.SVM_C_SVC,
                           C = 1 )

    trainData=np.array(training_set)
    responses=np.float32(training_labels)

    svm.train_auto(trainData, responses,None, None, params=svm_params)

    svm.save("svm.xml")
    
    tree = ET.parse('svm.xml')

root = tree.getroot()
SVs = root.getchildren()[0].getchildren()[-2].getchildren()[0] 
rho = float(root.getchildren()[0].getchildren()[-1].getchildren()[0].getchildren()[1].text )
svmvec = [float(x) for x in re.sub( '\s+', ' ', SVs.text ).strip().split(' ')]
svmvec.append(-rho)
pickle.dump(svmvec, open("svm.pickle", 'w')) 


img = cv2.imread(ppath+"01.tif", 0)
img = img[:1592, :3016]
hog = cv2.HOGDescriptor((32,64), (16,16), (8,8), (8,8), 9)
svm1 = pickle.load(open("svm.pickle"))

#print help(hog)


print len(svm1[:237440])

print np.array(svm1[:237440]).shape
print hog
print hog.getDescriptorSize

#hog.setSVMDetector( np.array(svm1[:237440]) )
#del svm1
#found, w = hog.detectMultiScale(img)

#print found

#cv2.imshow("found", found)
#cv2.waitkey(0)



