import cv2
import numpy as np
import math
#import matplotlib.pyplot as plt
import os
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score
#import pywt
#from matplotlib.pyplot import *
#from matplotlib import pyplot as plt
import pickle
from sklearn.decomposition import PCA
import argparse

path = "/home/chhavi/Downloads/sample"
colorhist = open("colorhist_meme_features.pkl", 'wb')
hog_features=[]
lbldata=[]
fnamelist=[]
for fpath,dic,files in os.walk(path):
     count = 0
     print(fpath)

     for fname in files:
         try:
             dpath = fpath + "/" + fname
             if "non_meme" in fpath:
                 lable = 0
             else:
                 lable = 1

             img = cv2.imread(dpath)
             width = int(img.shape[1] * 60 / 100)
             height = int(img.shape[0] * 60 / 100)
             dim = (width, height)
             # resize image
             resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
             chans = cv2.split(img)
             colors = ("b", "g", "r")
             features = []
             for (chan,color) in zip(chans, colors):
                hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
                # print(hist.shape)
                features.extend(hist)
                # print("fe",len(features))
             # pca = PCA(n_components=2)
             # vector = pca.fit_transform(features)
             vector =np.array(features).flatten()
             #fwrite.writelines(fname + "," + str(vector) + "," + str(lable) + "\n")
             hog_features.append(vector)
             lbldata.append(lable)
             fnamelist.append(fname.strip())
         except:
             print("error@",fname)
# pickle.dump((fnamelist,hog_features,lbldata),colorhist)
# labels = np.array(lbldata).reshape(len(lbldata), 1)
#
# clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#                            decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
#                            max_iter=-1, probability=False, random_state=None, shrinking=True,
#                            tol=0.001, verbose=False)
# hog_features = np.array(hog_features)
# print(hog_features.shape)
# print(labels.shape)
# data_frame = np.hstack((hog_features, labels))
# np.random.shuffle(data_frame)
# percentage = 80
# partition = int(len(hog_features) * percentage / 100)
# x_train, x_test = data_frame[:partition, :-1], data_frame[partition:, :-1]
# y_train, y_test = data_frame[:partition, -1:].ravel(), data_frame[partition:, -1:].ravel()
# clf.fit(x_train, y_train)
# y_pred = clf.predict(x_test)
# print("Accuracy: " + str(accuracy_score(y_test, y_pred)))
# print('\n')
# print(classification_report(y_test, y_pred))
