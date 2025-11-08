import cv2
import numpy as np
#from matplotlib import pyplot as plt
import os
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score
from sklearn.decomposition import PCA
import pickle

def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

def lbp_calculated_pixel(img, x, y):
    '''
     64 | 128 |   1
    ----------------
     32 |   0 |   2
    ----------------
     16 |   8 |   4    
    '''    
    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x-1, y+1))     # top_right
    val_ar.append(get_pixel(img, center, x, y+1))       # right
    val_ar.append(get_pixel(img, center, x+1, y+1))     # bottom_right
    val_ar.append(get_pixel(img, center, x+1, y))       # bottom
    val_ar.append(get_pixel(img, center, x+1, y-1))     # bottom_left
    val_ar.append(get_pixel(img, center, x, y-1))       # left
    val_ar.append(get_pixel(img, center, x-1, y-1))     # top_left
    val_ar.append(get_pixel(img, center, x-1, y))       # top
    
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val    

def show_output(output_list):
    output_list_len = len(output_list)
    figure = plt.figure()
    for i in range(output_list_len):
        current_dict = output_list[i]
        current_img = current_dict["img"]
        current_xlabel = current_dict["xlabel"]
        current_ylabel = current_dict["ylabel"]
        current_xtick = current_dict["xtick"]
        current_ytick = current_dict["ytick"]
        current_title = current_dict["title"]
        current_type = current_dict["type"]
        current_plot = figure.add_subplot(1, output_list_len, i+1)
        if current_type == "gray":
            current_plot.imshow(current_img, cmap = plt.get_cmap('gray'))
            current_plot.set_title(current_title)
            current_plot.set_xticks(current_xtick)
            current_plot.set_yticks(current_ytick)
            current_plot.set_xlabel(current_xlabel)
            current_plot.set_ylabel(current_ylabel)
        elif current_type == "histogram":
            current_plot.plot(current_img, color = "black")
            current_plot.set_xlim([0,260])
            current_plot.set_title(current_title)
            current_plot.set_xlabel(current_xlabel)
            current_plot.set_ylabel(current_ylabel)            
            ytick_list = [int(i) for i in current_plot.get_yticks()]
            current_plot.set_yticklabels(ytick_list,rotation = 90)

    plt.show()
    
def main():
    path = "/home/chhavi/Downloads/sample"
    #sift_org = open("lbp_meme.pkl", 'wb')
    lbp = open('lbp_meme_features1.pkl', 'wb')
    #sift_pca = open('lbp_pca2_org_meme.pkl', 'wb')
    lbl=[]
    dataset=[]
    fnamelist = []
    orgsift_features = []
    pcasift_features = []
    for fpath,fdic,files in os.walk(path):
        for fname in files:
            try:
                print("fname",fname)
                dpath = fpath + "/" + fname
                if "non_meme" in fpath:
                    lable = 0
                else:
                    lable = 1

                img_bgr = cv2.imread(dpath)
                height, width, channel = img_bgr.shape
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

                img_lbp = np.zeros((height, width,3), np.uint8)
                #print(img_lbp)
                for i in range(0, height):
                    for j in range(0, width):
                         img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
                hist_lbp = cv2.calcHist([img_lbp], [0], None, [256], [0, 256])
                # lbp=hist_lbp
                # print(lbp.shape)
                # break
                hist_lbp = hist_lbp.reshape(1, 256)
                # pca = PCA(n_components=2)
                # hist_lbp= pca.fit_transform(lbp)
                # hist_lbp =hist_lbp.reshape(1, hist_lbp.shape[0]*hist_lbp.shape[1])
                #fwrite.writelines(fname + "," + str(hist_lbp) + "," + str(lable) + "\n")
                dataset.append(hist_lbp)
                lbl.append(lable)
                fnamelist.append(fname.strip())
                pcasift_features.append(hist_lbp)

            except:
                print("error@",fname)
    pickle.dump((fnamelist, dataset, lbl), lbp)
    # pickle.dump((fnamelist, orgsift_features, lbl), sift_org)
    # pickle.dump((fnamelist, pcasift_features, lbl), sift_pca)
    # print("--",dataset)
    # print("-",lbl)
    # labels = np.array(lbl).reshape(len(lbl), 1)
    # print(labels)
    # clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    #               decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    #               max_iter=-1, probability=False, random_state=None, shrinking=True,
    #               tol=0.001, verbose=False)
    # hog_features = np.array(dataset)
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




if __name__ == '__main__':
    main()
