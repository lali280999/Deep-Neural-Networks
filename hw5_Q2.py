from keras.applications.resnet import ResNet50
import numpy as np
from keras.models import Model
from keras.layers import Dense,Flatten
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

trainY = np.load("/home/girivaasan/Documents/DL/Dataset/facesAndAges/ages.npy")
trainX = np.load("/home/girivaasan/Documents/DL/Dataset/facesAndAges/faces.npy")/255

def shuffle(x, y):
    n = y.shape[0]
    shuffle_idx = np.random.permutation(n)
    Xnew = x[shuffle_idx]
    Ynew = y[shuffle_idx]
    return Xnew, Ynew

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

trainX, trainY = shuffle(trainX, trainY)

data = np.stack((trainX,)*3, axis = -1)

X_train, X_test, y_train, y_test = train_test_split(data,trainY,test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size=0.1, random_state=42)

rnetmodel = ResNet50(input_shape=[48,48,3],weights='imagenet', include_top=False)

x=Flatten()(rnetmodel.output)
x=Dense(1024,activation='relu')(x)
x=Dense(512,activation='relu')(x)
prediction=Dense(1)(x)
final_model=Model(inputs=rnetmodel.input,outputs=prediction)
final_model.summary()

final_model.compile(optimizer = "adam", loss = root_mean_squared_error, metrics =["accuracy"])

checkpointer = ModelCheckpoint(filepath='final_model.weights.best.hdf5', verbose=1, save_best_only=True)
final_model.fit(X_train,y_train, batch_size=32, epochs=20, validation_data=(X_val, y_val), callbacks=[checkpointer])

final_model.load_weights('final_model.weights.best.hdf5')
test_score = final_model.evaluate(X_test, y_test, verbose=0)
print("Test Score:", test_score)