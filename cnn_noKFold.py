import os
import pickle
import json
import imageio
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras import backend as K
import skimage
from sklearn.model_selection import train_test_split, KFold
import pydicom

os.environ["CUDA_VISIBLE_DEVICES"]="0"

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def matthews_correlation(y_true, y_pred):
    '''Calculates the Matthews correlation coefficient measure for quality
    of binary classification problems.
    '''
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

X = []
y = []

dir = "./Data/Patches/NoPlaque"
for filename in os.listdir(dir):
    img = np.array(imageio.imread(os.path.join(dir, filename)))[:,:,0:1]
    #img = np.load(os.path.join(dir, filename))[:,:,0]
    X.append(img)
    y.append([1,0,0])

dir = "./Data/Patches/Plaque/Aug/cal"
for filename in os.listdir(dir):
    img = np.array(imageio.imread(os.path.join(dir, filename)))[:,:,0:1]
    #img = np.load(os.path.join(dir, filename))[:,:,0]
    X.append(img)
    y.append([0,1,0])

dir = "./Data/Patches/Plaque/Aug/fibrous"
for filename in os.listdir(dir):
    img = np.array(imageio.imread(os.path.join(dir, filename)))[:,:,0:1]
    #img = np.load(os.path.join(dir, filename))[:,:,0]
    X.append(img)
    y.append([0,0,1])


X = np.array(X)
y = np.array(y)

Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.25, random_state=42)

classifier = Sequential()
classifier.add(Conv2D(64, (5, 5), input_shape = (64, 64, 1), activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(MaxPooling2D(pool_size = (4, 4)))
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(MaxPooling2D(pool_size = (4, 4)))
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'sigmoid'))
classifier.add(Dropout(0.3))
classifier.add(Dense(units = 32, activation = 'sigmoid'))
classifier.add(Dense(units = 3, activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy', matthews_correlation])

classifier.summary()
es = EarlyStopping(patience=6, restore_best_weights = True) 

history2 = classifier.fit(Xtr, ytr,
				callbacks =  [es],
				epochs = 75,
				batch_size = 8,
				validation_data = (Xts, yts),
				class_weight = "balanced")
classifier.evaluate(Xts, yts)
classifier.save("./Models/model/noKfold_MCC.h5")
json.dump(history2.history, open("./Models/JSON/noKfold_MCC.json", "w"))
