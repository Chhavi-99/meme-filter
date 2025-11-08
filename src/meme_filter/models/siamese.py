import pickle
import keras as keras
from keras.optimizers import Adam
from tensorflow.python.keras.callbacks import *
from sklearn.metrics import *
from keras.utils import to_categorical
from keras.regularizers import l2
def initialize_bias(shape, name=None, dtype=float ):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)

def initialize_weights(shape, name=None, dtype=float):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)

emb= open('/home/chhavi/Downloads/VGG16_meme_v3.pkl','rb')
# emb=open('emb_emotionfunny.pkl','rb')
name,feature,lables= pickle.load(emb)
text=open('sentenceEncode_meme.pkl','rb')
text_name, text_feature, text_lbl=pickle.load(text)
print(type(name))
feature1=[]
txt_feature=[]
nlbl=[]
# for i in range(0,len(text_feature)):
#     fname = text_name[i]
#     indx = name.index(fname)
#     feature1.append(feature[indx])
#     nlbl.append(lables[indx])

for i in text_name:
    for j in name:
        if str(i).strip() in str(j).strip():
            # print(i)
            inx = name.index(str(i).strip())
            inxt = text_name.index(str(i).strip())
            feature1.append(feature[inx])
            # vec =np.array(text_feature[inxt])
            txt_feature.append(text_feature[inxt])
            nlbl.append(lables[inx])


trian_data = np.asarray(feature1)
lables=np.array(nlbl).reshape(len(nlbl),1)
trian_data2 = np.asarray(txt_feature)
text_lbl= np.array(text_lbl).reshape(len(text_lbl),1)

z=list(zip(trian_data,trian_data2,lables))
np.random.shuffle(z)
trian_data,train_data2,lables=zip(*z)

trian_data=np.array(trian_data)
train_data2 = np.array(train_data2)
lables=np.array(lables)

train_x =trian_data[:4500]
train_x2 = trian_data2[:4500]
train_y = to_categorical(lables[:4500])
print(train_x.shape)
test_x = trian_data[4501:]
test_x2 = trian_data2[4501:]
sample = lables[4501:]
test_lables = np.asarray(lables[4501:])
test_lables= to_categorical(test_lables)


i2  = keras.Input(shape=(4096,), dtype=float)
# c1 = keras.layers.Conv1D(15, 2, activation='relu')(i2)
# c2 = keras.layers.MaxPooling1D(2)(c1)
# f1 = keras.layers.Flatten()(c2)
d1= keras.layers.Dense(2000,activation='sigmoid',
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer=initialize_weights,bias_initializer=initialize_bias)(i2)
d1= keras.layers.Dense(500,activation='softmax',
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer=initialize_weights,bias_initializer=initialize_bias)(d1)

model2= keras.Input(shape=(512,), dtype=float)
d2 = keras.layers.Dense(500, activation='softmax',
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer=initialize_weights,bias_initializer=initialize_bias)(model2)
L1_layer = keras.layers.Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
L1_distance = L1_layer([d1, d2])
# concat_layers = keras.layers.concatenate([model1, model2])
# layer = keras.layers.Dense(100)(L1_distance)

layer1= keras.layers.Dense(2,activation='softmax',bias_initializer=initialize_bias)(L1_distance)
# outlayer = keras.layers.Activation('softmax')(layer1)
model = keras.Model(inputs=[i2, model2], outputs=layer1)



model.summary()
model.compile(Adam(lr = 0.00006),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit([train_x,train_x2], train_y, epochs=40, batch_size=50, validation_split=.1, verbose=1)

y_pred1 = model.predict([test_x,test_x2])
get_5rd_layer_output = K.function([[model.layers[0].input],[model.layers[2].input]],
                                      [model.layers[5].output])
layerout =get_5rd_layer_output([test_x,test_x2])[0]
print("---",layerout.shape)
print("---",layerout)
memfile = open("memefile1.csv","a")
for  idx in range(0,len(layerout)):
    print(idx,"--",sample[idx],"---",np.mean(layerout[idx]))
    memfile.writelines(str(idx)+","+str(sample[idx])+","+str(np.mean(layerout[idx]))+"\n")
memfile.close()
y_pred = np.argmax(y_pred1, axis=1)
print(y_pred)
scores1 = model.evaluate([test_x,test_x2],test_lables,verbose=0)

print("%s: %.2f%%" % (model.metrics_names[1], scores1[1]*100))
# Print f1, precision, and recall scores
print("p", precision_score(sample, y_pred,average='weighted'))
print("r", recall_score(sample, y_pred, average='weighted'))
print("f1", f1_score(sample, y_pred, average='weighted'))



