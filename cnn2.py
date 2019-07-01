import os
import pickle
import json
import cv2
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

def get_model():
    classifier = Sequential()
    classifier.add(Conv2D(64, (5, 5), input_shape = (64, 64, 1), activation = 'relu'))
    classifier.add(Dropout(0.4))
    classifier.add(MaxPooling2D(pool_size = (4, 4)))
    classifier.add(Conv2D(32, (4, 4), activation = 'relu'))
    classifier.add(Dropout(0.2))
    classifier.add(MaxPooling2D(pool_size = (4, 4)))
    classifier.add(Flatten())
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dropout(0.3))
    classifier.add(Dense(units = 32, activation = 'relu'))
    classifier.add(Dense(units = 3, activation = 'softmax'))
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy', matthews_correlation])
    return classifier

X = []
y = []


dir = "./Data/Patches/NoPlaque"
for filename in os.listdir(dir)[:3790]:
    img = np.array(cv2.imread(os.path.join(dir, filename)))[:,:,0:1]
    #img = np.load(os.path.join(dir, filename))[:,:,0]
    X.append(img)
    y.append([1,0,0])

dir = "./Data/Patches/Plaque/Aug/cal"
for filename in os.listdir(dir):
    img = np.array(cv2.imread(os.path.join(dir, filename)))[:,:,0:1]
    #img = np.load(os.path.join(dir, filename))[:,:,0]
    X.append(img)
    y.append([0,1,0])

dir = "./Data/Patches/Plaque/Aug/fibrous"
for filename in os.listdir(dir):
    img = np.array(cv2.imread(os.path.join(dir, filename)))[:,:,0:1]
    #img = np.load(os.path.join(dir, filename))[:,:,0]
    X.append(img)
    y.append([0,0,1])


X = np.array(X)
y = np.array(y)
print(X.shape)
print(X[0].shape)
seeds = range(1,40, 10)
for i, seed in enumerate(seeds):
    Xtr, Xts, ytr, yts = train_test_split(np.copy(X), np.copy(y), test_size=0.25, random_state=seed)
    classifier = get_model()
    classifier.summary()

    print("Run: ", i)
    es = EarlyStopping(patience=5, restore_best_weights = True)
    history = classifier.fit(Xtr, ytr,
					callbacks = [es],
					epochs = 25,
					batch_size = 32,
					validation_data = (Xts, yts),
					class_weight = "balanced")
    classifier.evaluate(Xts, yts)
    classifier.save("./Models/model/newSeed" + str(seed) + ".h5")
    json.dump(history.history, open("./Models/JSON/newSeed" + str(seed) + ".json", "w"))
