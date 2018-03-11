# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 16:37:18 2018

@author: NishitP
"""

import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
from sklearn.metrics import accuracy_score


# Reading the dataset
mndata = MNIST('./')

#untouched train set
images, labels = mndata.load_training()

#rescaling images for train
ScaleImages, ScaleLabels = mndata.load_training()

# untouched test set
images_test, labels_test = mndata.load_testing()

# rescaling images for test 
Scale_images_test, Scale_labels_test = mndata.load_testing()


# Custom functons 
def resizeT20(imgdata):
    imarray = np.asfarray(imgdata).reshape((28,28))
    step1Array= imarray[:,~np.all(imarray == 0, axis=0)]
    finalArray=step1Array[~np.all(step1Array == 0, axis=1)]
    img= Image.fromarray(finalArray)
    new_image = img.resize((20, 20))
    resized=list(new_image.getdata())
    #resizedT20 = np.asfarray(resized).reshape((20,20))
    return resized  

def printOriginalImage(imgdata):        
    imarray = np.asfarray(imgdata).reshape((28,28))
    plt.imshow(imarray, cmap='Greys', interpolation='None')    
    
def printT20Image(imgdata):        
    imarray = np.asfarray(imgdata).reshape((20,20))
    plt.imshow(imarray, cmap='Greys', interpolation='None') 
  
    
def thresholdAndResize(inputList):
    resizedImages=list()
    #Thresholding the image
    for lst in inputList:
        for ind, item in enumerate(lst):
             if lst[ind] > 128:
                 lst[ind] = 255
             else:
                 lst[ind] = 0
        resizedImages.append(resizeT20(lst))
    return resizedImages


#printOriginalImage(images[0])
#printT20Image(resizedImages[0])

resizedTrainImages = thresholdAndResize(ScaleImages)
resizedTestImages = thresholdAndResize(Scale_images_test)
    
index=2
printOriginalImage(images[index])
print(1, end=' ')
printT20Image(resizedTrainImages[index])
print(1, end=' ')
printT20Image(resizedTestImages[index])


#PART A - NAIVE BAYES
labels_arr = np.asarray(labels)
images_arr = np.asarray(images)

labels_arr_str = np.asarray(labels)
images_arr_str = np.asarray(resizedTrainImages)

labels_arr.shape
images_arr.shape

# untouched gaussian distribution
classifier = GaussianNB()
classifier.fit(images_arr,labels_arr)

predicted = classifier.predict(np.asarray(images_test))
accuracy_score(predicted,np.asarray(labels_test))

#stretched box guassian distribution
classifier_str = GaussianNB()
classifier_str.fit(images_arr_str,labels_arr_str)

predicted_str = classifier_str.predict(np.asarray(resizedTestImages))
accuracy_score(predicted_str, np.asarray(Scale_labels_test))

np.asarray(resizedTestImages).shape
np.asarray(Scale_labels_test).shape


# Untouched Bernoulli distribution
classifier_ber = BernoulliNB()
classifier_ber.fit(images_arr,labels_arr)

predicted_ber = classifier_ber.predict(np.asarray(images_test))
accuracy_score(predicted_ber, np.asarray(labels_test))


# Stretched Bernoulli distribution
classifier_ber_str = BernoulliNB()
classifier_ber_str.fit(images_arr_str,labels_arr_str)

predicted_ber_str = classifier_ber_str.predict(np.asarray(resizedTestImages))
accuracy_score(predicted_ber_str,np.asarray(Scale_labels_test))


def printImage(imgdata):
    #f = plt.figure(figsize=(15,15));        
    imarray = np.asfarray(imgdata).reshape((28,28))
    #imarray.delete(imarray, imarray.nonzero((imarray==0).sum(axis=0) > 5), axis=1)
    #imarray.tolist.delete(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    #shortArray=imarray[:, np.sum(imarray == 0, axis=0) == 0 ]
    shortArray= imarray[:,~np.all(imarray == 0, axis=0)]
    shortenedFinal=shortArray[~np.all(shortArray == 0, axis=1)]
    #shortArray= imarray[~np.all(imarray == 0, axis=0)]
    #plt.subplot(28,28,1)
    #plt.subplots_adjust(hspace=0.5)
    #plt.title("Label is " + str(labels[count-2]))

    #resize(shortenedFinal, (20, 20), mode='reflect').shape(20, 20)    
    plt.imshow(shortenedFinal, cmap='Greys', interpolation='None')
    #print(shortArray)
    return shortenedFinal

#shortArray= printImage(scaled[0])




## PART B - RANDOM FOREST
## Untouched raw pixels
#trees = 10, depth = 4
classifier_rf_10_4 = RandomForestClassifier(max_depth=4,n_estimators=10, random_state=0)
classifier_rf_10_4.fit(images_arr,labels_arr)

predicted_rf_10_4 = classifier_rf_10_4.predict(np.asarray(images_test))
accuracy_score(predicted_rf_10_4, np.asarray(labels_test))
# 73.9%

#trees = 10 , depth = 8 
classifier_rf_10_8 = RandomForestClassifier(max_depth=8,n_estimators=10, random_state=0)
classifier_rf_10_8.fit(images_arr,labels_arr)

predicted_rf_10_8 = classifier_rf_10_8.predict(np.asarray(images_test))
accuracy_score(predicted_rf_10_8, np.asarray(labels_test))
# 90%

#trees = 10 , depth = 16 
classifier_rf_10_16 = RandomForestClassifier(max_depth=16,n_estimators=10, random_state=0)
classifier_rf_10_16.fit(images_arr,labels_arr)

predicted_rf_10_16 = classifier_rf_10_16.predict(np.asarray(images_test))
accuracy_score(predicted_rf_10_16, np.asarray(labels_test))
# 94.7%

#trees = 20 , depth = 4 
classifier_rf_20_4 = RandomForestClassifier(max_depth=4, n_estimators=20, random_state=0)
classifier_rf_20_4.fit(images_arr,labels_arr)

predicted_rf_20_4 = classifier_rf_20_4.predict(np.asarray(images_test))
accuracy_score(predicted_rf_20_4,np.asarray(labels_test))
# 79.4%

#trees = 20 , depth = 8
classifier_rf_20_8 = RandomForestClassifier(max_depth=8, n_estimators=20, random_state=0)
classifier_rf_20_8.fit(images_arr,labels_arr)

predicted_rf_20_8 = classifier_rf_20_8.predict(np.asarray(images_test))
accuracy_score(predicted_rf_20_8, np.asarray(labels_test))
# 91.5%

#trees = 20 , depth = 16
classifier_rf_20_16 = RandomForestClassifier(max_depth=16, n_estimators=20, random_state=0)
classifier_rf_20_16.fit(images_arr,labels_arr)

predicted_rf_20_16 = classifier_rf_20_16.predict(np.asarray(images_test))
accuracy_score(predicted_rf_20_16, np.asarray(labels_test))
# 95.9%

#trees = 30, depth = 4
classifier_rf_30_4 = RandomForestClassifier(max_depth=4,n_estimators=30, random_state=0)
classifier_rf_30_4.fit(images_arr,labels_arr)

predicted_rf_30_4 = classifier_rf_30_4.predict(np.asarray(images_test))
accuracy_score(predicted_rf_30_4, np.asarray(labels_test))
# 80.1%

#trees = 30, depth = 8
classifier_rf_30_8 = RandomForestClassifier(max_depth=8,n_estimators=30, random_state=0)
classifier_rf_30_8.fit(images_arr,labels_arr)

predicted_rf_30_8 = classifier_rf_30_8.predict(np.asarray(images_test))
accuracy_score(predicted_rf_30_8,np.asarray(labels_test))
# 92.2%

#trees = 30, depth = 16
classifier_rf_30_16 = RandomForestClassifier(max_depth=16,n_estimators=30, random_state=0)
classifier_rf_30_16.fit(images_arr,labels_arr)

predicted_rf_30_16 = classifier_rf_30_16.predict(np.asarray(images_test))
accuracy_score(predicted_rf_30_16, np.asarray(labels_test))
# 96.2%


##Streched bounding box
#trees = 10, depth = 4
clf_rf_10_4 = RandomForestClassifier(max_depth=4,n_estimators=10, random_state=0)
clf_rf_10_4.fit(images_arr_str,labels_arr_str)

predicted_rf_10_4 = clf_rf_10_4.predict(np.asarray(resizedTestImages))
accuracy_score(predicted_rf_10_4, np.asarray(Scale_labels_test))
# 72.13%

#trees = 10 , depth = 8 
clf_rf_10_8 = RandomForestClassifier(max_depth=8,n_estimators=10, random_state=0)
clf_rf_10_8.fit(images_arr_str,labels_arr_str)

predicted_rf_10_8 = clf_rf_10_8.predict(np.asarray(resizedTestImages))
np.mean(predicted_rf_10_8 == np.asarray(Scale_labels_test))
accuracy_score(predicted_rf_10_8, np.asarray(Scale_labels_test))
# 89.23%

#trees = 10 , depth = 16 
clf_rf_10_16 = RandomForestClassifier(max_depth=16,n_estimators=10, random_state=0)
clf_rf_10_16.fit(images_arr_str,labels_arr_str)

predicted_rf_10_16 = clf_rf_10_16.predict(np.asarray(resizedTestImages))
accuracy_score(predicted_rf_10_16, np.asarray(Scale_labels_test))
# 94.8%

#trees = 20 , depth = 4 
clf_rf_20_4 = RandomForestClassifier(max_depth=4, n_estimators=20, random_state=0)
clf_rf_20_4.fit(images_arr_str,labels_arr_str)

predicted_rf_20_4 = clf_rf_20_4.predict(np.asarray(resizedTestImages))
accuracy_score(predicted_rf_20_4, np.asarray(Scale_labels_test))
# 74.9%

#trees = 20 , depth = 8
clf_rf_20_8 = RandomForestClassifier(max_depth=8, n_estimators=20, random_state=0)
clf_rf_20_8.fit(images_arr_str,labels_arr_str)

predicted_rf_20_8 = clf_rf_20_8.predict(np.asarray(resizedTestImages))
accuracy_score(predicted_rf_20_8, np.asarray(Scale_labels_test))
# 90.5%

#trees = 20 , depth = 16
clf_rf_20_16 = RandomForestClassifier(max_depth=16, n_estimators=20, random_state=0)
clf_rf_20_16.fit(images_arr_str,labels_arr_str)

predicted_rf_20_16 = clf_rf_20_16.predict(np.asarray(resizedTestImages))
accuracy_score(predicted_rf_20_16, np.asarray(Scale_labels_test))
# 95.86%

#trees = 30, depth = 4
clf_rf_30_4 = RandomForestClassifier(max_depth=4,n_estimators=30, random_state=0)
clf_rf_30_4.fit(images_arr_str,labels_arr_str)

predicted_rf_30_4 = clf_rf_30_4.predict(np.asarray(resizedTestImages))
accuracy_score(predicted_rf_30_4, np.asarray(Scale_labels_test))
# 75.88%

#trees = 30, depth = 8
clf_rf_30_8 = RandomForestClassifier(max_depth=8,n_estimators=30, random_state=0)
clf_rf_30_8.fit(images_arr_str,labels_arr_str)

predicted_rf_30_8 = clf_rf_30_8.predict(np.asarray(resizedTestImages))
accuracy_score(predicted_rf_30_8, np.asarray(Scale_labels_test))
# 90.73%

#trees = 30, depth = 16
clf_rf_30_16 = RandomForestClassifier(max_depth=16,n_estimators=30, random_state=0)
clf_rf_30_16.fit(images_arr_str,labels_arr_str)

predicted_rf_30_16 = clf_rf_30_16.predict(np.asarray(resizedTestImages))
accuracy_score(predicted_rf_30_16, np.asarray(Scale_labels_test))
# 96.3%


