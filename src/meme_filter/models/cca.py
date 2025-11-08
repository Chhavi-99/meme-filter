import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA
import pickle
from sklearn.model_selection import train_test_split
from mpl_toolkits import mplot3d
# def f(x, y):
#     return np.sqrt(np.square(x) + np.square(y))
#
#
# x = np.linspace(0, 1, 100)
# y = np.linspace(0, 1, 100)
#
#
# X, Y = np.meshgrid(x, y)
# print(X.shape)
# print(Y)
#
#
# Z = f(X, Y)
#
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.contour3D(X, Y, Z, 50, cmap='coolwarm')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.set_title('3D contour')
# plt.show()
#
# #
# # fig = plt.figure()
# # ax = plt.axes(projection='3d')
# # ax.contour3D(X, Y, Z, 50, cmap="viridis")
# # ax.set_xlabel('x')
# # ax.set_ylabel('y')
# # ax.set_zlabel('z')
# # ax.set_title('3D contour')
# # plt.show()
# # xlist = [0.9,0.8,0.1]
# # ylist = [0.5,0.3,.45]
# # X, Y = np.meshgrid(xlist, ylist)
# # print(X,Y)
# # Z = np.sqrt(X**2 + Y**2)
# # print(Z)
# # fig,ax=plt.subplots(1,1)
# # cp = ax.contourf(X, Y, Z)
# # fig.colorbar(cp) # Add a colorbar to a plot
# # ax.set_title('Filled Contours Plot')
# # #ax.set_xlabel('x (cm)')
# # ax.set_ylabel('y (cm)')
# # plt.show()

emb= open('VGG16_meme_v3 (1).pkl','rb')
# emb=open('emb_emotionfunny.pkl','rb')
name,feature,imglbl= pickle.load(emb)
text=open('sentenceEncode_meme.pkl','rb')
text_name, text_feature, text_lbl=pickle.load(text)
print(len(name),len(imglbl))
print(len(text_name),len(text_feature))
f= open('cca_meme_vgg_v1.pkl','wb')
feature1=[]
txt_feature=[]
nlbl=[]
count=0
img_name=[]
for i in text_name:
    for j in name:
        try:
            if str(i).strip() in str(j).strip():

                inx = name.index(str(i).strip())
                inxt = text_name.index(str(i).strip())
                # print(text_lbl[inxt] )
                # if str(text_lbl[inxt]).strip() in "1":
                feature1.append(feature[inx])
                txt_feature.append(text_feature[inxt])
                nlbl.append(text_lbl[inxt])
                img_name.append(str(i).strip())
        except:
            print("error @",i)

print(len(feature1),len(txt_feature),len(nlbl))

trainx =np.array(feature1)
trainy =np.array(txt_feature)
X_train =trainx[0:4000]
Y_train= trainy[0:4000]
X_test =trainx[4001:]
Y_test = trainy[4001:]
X_train,X_test,Y_train,Y_test,train_lbl,test_lbl=train_test_split(
    feature1, txt_feature, nlbl, test_size=.2, random_state=42)

cca = CCA(n_components=1)
cca.fit(X_train, Y_train)
X_train_r, Y_train_r = cca.transform(X_train, Y_train)
X_test_r, Y_test_r = cca.transform(X_test, Y_test)
print(X_test_r.shape)

print(Y_test_r.shape)

# print(np.corrcoef(X_train_r,Y_train_r)[0,1])

# for i in range(0,len(X_test_r)):
#     print(test_lbl[i],",",np.corrcoef(X_test_r[i],Y_test_r[i])[0,1])

print(X_test_r.shape)
fopen = open("cca_1.pkl","wb")
cca_meme=[]
for i in range(0,len(X_test_r)):
    # print("---",X_test_r[i])
    # print("---",Y_test_r[i])
    z = np.concatenate((X_test_r[i],Y_test_r[i]))
    cca_meme.append(z)
pickle.dump((img_name,cca_meme,nlbl),fopen)





# # 1) On diagonal plot X vs Y scores on each components
# plt.figure(figsize=(12, 8))
# plt.subplot(221)
# plt.scatter(X_train_r[:, 0], Y_train_r[:, 0], label="train",
#             marker="o", c="b", s=25)
# plt.scatter(X_test_r[:, 0], Y_test_r[:, 0], label="test",
#             marker="o", c="r", s=25)
# plt.xlabel("x scores")
# plt.ylabel("y scores")
# plt.title('Comp. 1: X vs Y (test corr = %.2f)' %
#           np.corrcoef(X_test_r[:, 0], Y_test_r[:, 0])[0, 1])
# plt.xticks(())
# plt.yticks(())
# plt.legend(loc="best")
#
# plt.subplot(224)
# plt.scatter(X_train_r[:, 1], Y_train_r[:, 1], label="train",
#             marker="o", c="b", s=25)
# plt.scatter(X_test_r[:, 1], Y_test_r[:, 1], label="test",
#             marker="o", c="r", s=25)
# plt.xlabel("x scores")
# plt.ylabel("y scores")
# plt.title('Comp. 2: X vs Y (test corr = %.2f)' %
#           np.corrcoef(X_test_r[:, 1], Y_test_r[:, 1])[0, 1])
# plt.xticks(())
# plt.yticks(())
# plt.legend(loc="best")
#
# # 2) Off diagonal plot components 1 vs 2 for X and Y
# plt.subplot(222)
# plt.scatter(X_train_r[:, 0], X_train_r[:, 1], label="train",
#             marker="*", c="b", s=50)
# plt.scatter(X_test_r[:, 0], X_test_r[:, 1], label="test",
#             marker="*", c="r", s=50)
# plt.xlabel("X comp. 1")
# plt.ylabel("X comp. 2")
# plt.title('X comp. 1 vs X comp. 2 (test corr = %.2f)'
#           % np.corrcoef(X_test_r[:, 0], X_test_r[:, 1])[0, 1])
# plt.legend(loc="best")
# plt.xticks(())
# plt.yticks(())

# plt.subplot(223)
# plt.scatter(Y_train_r[:, 0], Y_train_r[:, 1], label="train",
#             marker="*", c="b", s=50)
# plt.scatter(Y_test_r[:, 0], Y_test_r[:, 1], label="test",
#             marker="*", c="r", s=50)
# plt.xlabel("Y comp. 1")
# plt.ylabel("Y comp. 2")
# plt.title('Y comp. 1 vs Y comp. 2 , (test corr = %.2f)'
#           % np.corrcoef(Y_test_r[:, 0], Y_test_r[:, 1])[0, 1])
# plt.legend(loc="best")
# plt.xticks(())
# plt.yticks(())
# plt.show()
