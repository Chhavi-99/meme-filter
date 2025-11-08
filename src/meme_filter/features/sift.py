import cv2
import numpy as np
import math
#import matplotlib.pyplot as plt
import os
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score
from sklearn.decomposition import PCA
import pickle

# class Hog_descriptor():
#     def __init__(self, img, cell_size=16, bin_size=8):
#         self.img = img
#         self.img = np.sqrt(img / float(np.max(img)))
#         self.img = self.img * 255
#         self.cell_size = cell_size
#         self.bin_size = bin_size
#         self.angle_unit = 360 / self.bin_size
#         assert type(self.bin_size) == int, "bin_size should be integer,"
#         assert type(self.cell_size) == int, "cell_size should be integer,"
#         #assert type(self.angle_unit) == int, "bin_size should be divisible by 360"
#
#     def extract(self):
#         height, width = self.img.shape
#         gradient_magnitude, gradient_angle = self.global_gradient()
#         gradient_magnitude = abs(gradient_magnitude)
#         cell_gradient_vector = np.zeros((int(height / self.cell_size), int(width / self.cell_size), self.bin_size))
#         for i in range(cell_gradient_vector.shape[0]):
#             for j in range(cell_gradient_vector.shape[1]):
#                 cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
#                                  j * self.cell_size:(j + 1) * self.cell_size]
#                 cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
#                              j * self.cell_size:(j + 1) * self.cell_size]
#                 cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)
#
#         hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)
#         hog_vector = []
#         for i in range(cell_gradient_vector.shape[0] - 1):
#             for j in range(cell_gradient_vector.shape[1] - 1):
#                 block_vector = []
#                 block_vector.extend(cell_gradient_vector[i][j])
#                 block_vector.extend(cell_gradient_vector[i][j + 1])
#                 block_vector.extend(cell_gradient_vector[i + 1][j])
#                 block_vector.extend(cell_gradient_vector[i + 1][j + 1])
#                 mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
#                 magnitude = mag(block_vector)
#                 if magnitude != 0:
#                     normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
#                     block_vector = normalize(block_vector, magnitude)
#                 hog_vector.append(block_vector)
#         return hog_vector, hog_image
#
#     def global_gradient(self):
#         gradient_values_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
#         gradient_values_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)
#         gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
#         gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
#         return gradient_magnitude, gradient_angle
#
#     def cell_gradient(self, cell_magnitude, cell_angle):
#         orientation_centers = [0] * self.bin_size
#         for i in range(cell_magnitude.shape[0]):
#             for j in range(cell_magnitude.shape[1]):
#                 gradient_strength = cell_magnitude[i][j]
#                 gradient_angle = cell_angle[i][j]
#                 min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
#                 orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
#                 orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
#         return orientation_centers
#
#     def get_closest_bins(self, gradient_angle):
#         idx = int(gradient_angle / self.angle_unit)
#         mod = gradient_angle % self.angle_unit
#         if idx == self.bin_size:
#             return idx - 1, (idx) % self.bin_size, mod
#         return idx, (idx + 1) % self.bin_size, mod
#
#     def render_gradient(self, image, cell_gradient):
#         cell_width = self.cell_size / 2
#         max_mag = np.array(cell_gradient).max()
#         for x in range(cell_gradient.shape[0]):
#             for y in range(cell_gradient.shape[1]):
#                 cell_grad = cell_gradient[x][y]
#                 cell_grad /= max_mag
#                 angle = 0
#                 angle_gap = self.angle_unit
#                 for magnitude in cell_grad:
#                     angle_radian = math.radians(angle)
#                     x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
#                     y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
#                     x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
#                     y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
#                     cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
#                     angle += angle_gap
#         return image
#


path = "/home/chhavi/Downloads/sample"
sift_org = open("sift_meme.pkl",'wb')
sift_pca200 = open('sift_meme_features2.pkl','wb')
sift_pca = open('sift_pca2_org_meme.pkl','wb')
hog_features=[]
lbldata=[]
fnamelist=[]
orgsift_features=[]
pcasift_features=[]
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

             img = cv2.imread(dpath, cv2.IMREAD_GRAYSCALE)

             width = int(img.shape[1] * 60 / 100)
             height = int(img.shape[0] * 60 / 100)
             dim = (width, height)
             # resize image
             resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
             sift = cv2.xfeatures2d.SIFT_create()
             #print(sift)
             keypoints_1, vector = sift.detectAndCompute(resized, None)
             #vector_pca= vector
             #vector = np.asarray(vector).reshape(np.asarray(vector).shape[0] * np.asarray(vector).shape[1])
             # orgsift_features.append(vector)
             # pca = PCA(n_components=2)
             # vector = pca.fit_transform(vector_pca)
             # vector = np.asarray(vector).reshape(np.asarray(vector).shape[0]*np.asarray(vector).shape[1])
             #fwrite.writelines(fname + "," + str(vector) + "," + str(lable) + "\n")
             # print(vector[0:100])
             hog_features.append(vector)
             fnamelist.append(fname.strip())
             #pcasift_features.append(vector)
             lbldata.append(lable)
         except:
                print("error@",fname)
pickle.dump((fnamelist,hog_features,lbldata),sift_pca200)
# pickle.dump((fnamelist,orgsift_features,lbldata), sift_org)
# pickle.dump((fnamelist,pcasift_features,lbldata), sift_pca)
# labels = np.array(lbldata).reshape(len(lbldata), 1)
# print(labels)
# clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#                    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
#                    max_iter=-1, probability=False, random_state=None, shrinking=True,
#                    tol=0.001, verbose=False)
# hog_features = np.array(hog_features)
# print(hog_features.shape)
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
