#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, RepeatedKFold
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.preprocessing import LabelEncoder
import pydicom
import random

output_txt = open("output.txt","w")
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# In[ ]:


def matthews_correlation(y_true, y_pred):
    '''Calculates the Matthews correlation coefficient measure for quality
    of binary classification problems.
    
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
    '''
    return mcc(y_true, y_pred)


# In[ ]:


def calc_mcc(yval, yval_rk):
    ycat = np.zeros(yval_rk.shape[0])
    x = 0
    for val in yval_rk:
        index = np.argmax(val)
        ycat[x] = index
        x += 1
    yreal = np.zeros(ycat.shape)
    for i in range(yval.shape[0]):
        yreal[i] = np.argmax(yval[i])
    #print(yreal)
    #print(ycat)
    #print('MCC: ', str(mcc(yreal, ycat)))
    return mcc(yreal,ycat)


# In[ ]:


def get_model(ytr, Xtr):
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
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier

def bootstrap_ci(x, B=1000, alpha=0.05, seed=0):
    """Computes the (1-alpha) Bootstrap confidence interval
    from empirical bootstrap distribution of sample mean.

    The lower and upper confidence bounds are the (B*alpha/2)-th
    and B * (1-alpha/2)-th ordered means, respectively.
    For B = 1000 and alpha = 0.05 these are the 25th and 975th
    ordered means.
    """

    x_arr = np.ravel(x)

    if B < 2:
        raise ValueError("B must be >= 2")

    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be in [0, 1]")

    np.random.seed(seed)

    bmean = np.empty(B, dtype=np.float)
    for b in range(B):
        idx = np.random.random_integers(0, x_arr.shape[0]-1, x_arr.shape[0])
        bmean[b] = np.mean(x_arr[idx])

    bmean.sort()
    lower = int(B * (alpha * 0.5))
    upper = int(B * (1 - (alpha * 0.5)))

    return (bmean[lower], bmean[upper])

def stats(mcc1, mcc2):
    total = 0
    inrange = 0
    outofrange = 0
    for num in mcc1:
        total += num
    avg = total/len(mcc1)
    min1 = np.amin(mcc1)
    max1 = np.amax(mcc1)
    conf_int = bootstrap_ci(mcc1)
    lower = conf_int[0]
    upper = conf_int[1]
    for num in mcc2:
        if(mcc2<upper and mcc2>lower):
            print('MCC is in confidence interval')
            inrange += 1
        else:
            print('MCC is NOT in confidence interval')
            outofrange += 1
    return avg, min1, max1, conf_int, inrange, outofrange

# In[ ]:


Xtemp = []
ytemp = []
X = []
y = []
index = []

count0 = 0
dir = "./Data/Patches/NoPlaque"
for filename in os.listdir(dir):
    img = np.array(cv2.imread(os.path.join(dir, filename)))[:,:,0:1]
    #img = np.load(os.path.join(dir, filename))[:,:,0]
    Xtemp.append(img)
    ytemp.append([1,0,0])
while(len(index) != 4000):
    r = random.randint(1,4000)
    if r not in index: 
        index.append(r)
for i in index:
    X.append(Xtemp[i])
    y.append(ytemp[i])
count0 = len(X)

count1 = 0
dir = "./Data/Patches/Plaque/Aug/cal"
for filename in os.listdir(dir):
    img = np.array(cv2.imread(os.path.join(dir, filename)))[:,:,0:1]
    #img = np.load(os.path.join(dir, filename))[:,:,0]
    X.append(img)
    y.append([0,1,0])
    count1 += 1

count2 = 0
dir = "./Data/Patches/Plaque/Aug/fibrous"
for filename in os.listdir(dir):
    img = np.array(cv2.imread(os.path.join(dir, filename)))[:,:,0:1]
    #img = np.load(os.path.join(dir, filename))[:,:,0]
    X.append(img)
    y.append([0,0,1])  
    count2 += 1


X = np.array(X)
y = np.array(y)
#print(X.shape)
#print(X[0].shape)
#seeds = range(2, 60, 10)
#for i, seed in enumerate(seeds):
#k_fold = KFold(len(y), n_folds=10, shuffle=True, random_state=0)
#rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)
#clf = get_model(ytr,Xtr)
#print(cross_val_score(clf, X, y, cv=rskf, n_jobs=1, scoring = 'accuracy'))
#clf.summary()
#history = classifier.fit(Xtr, ytr,
                #callbacks = [es],
                #epochs = 40,
                #batch_size = 32,
                #validation_data = (Xts, yts),
                #class_weight = class_weight)
#y_tr_r = classifier.predict(y_tr)
#print('MCC: ', str(mcc(y_tr, y_tr_r)))
#clf.evaluate(Xts, yts)
#clf.save("./Models/model/KFold" + str(train_index) + ".h5")
#json.dump(history.history, open("./Models/JSON/KFold" + str(train_index) + ".json", "w"))

weight_for_0 = 1. / count0
weight_for_1 = 1. / count1
weight_for_2 = 1. / count2
Xtr, Xts, ytr, yts = train_test_split(np.copy(X), np.copy(y), test_size=0.25)#, random_state=seed)
#ytr_categorical = keras.utils.to_categorical(ytr, num_classes = 3)
rskf = RepeatedKFold(n_splits=5, n_repeats=4)
mcc_all = []
mcc_tr = []
n = 0

for train_index, val_index in rskf.split(Xtr, ytr):
    #print("TRAIN:", train_index, "VAL:", val_index)
    Xtrk, Xval = Xtr[train_index], Xtr[val_index]
    ytrk, yval = ytr[train_index], ytr[val_index]

    mean = np.mean(Xtrk, axis=0)
    Xtrk = np.subtract(Xtrk,mean)
    #Xtsk = np.subtract(Xval,mean)
    std = np.std(Xtrk, axis=0)
    Xtrk = np.divide(Xtrk,std)
    #Xtsk = np.divide(Xval,std)
    classifier = get_model(ytrk, Xtrk)
    classifier.summary()

    print("Run: ", n)
    es = EarlyStopping(patience=5, restore_best_weights = True)
    class_weights = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2}
    
    history = classifier.fit(Xtrk, ytrk,
                    callbacks = [es],
                    epochs = 40,
                    batch_size = 32,
                    validation_data = (Xval, yval),
                    class_weight = class_weights)

    yval_rk = classifier.predict(Xval)
    mcc_tr.append(calc_mcc(yval, yval_rk))
    print('MCC ', str(n), ': ', str(calc_mcc(yval, yval_rk)))
    output_txt.write(('MCC ' + str(n) + ': ' + str(calc_mcc(yval, yval_rk))))
    classifier.evaluate(Xval, yval)
    classifier.save("./Models/model/KFoldNew_" + str(n) + ".h5")
    json.dump(history.history, open("./Models/JSON/KFoldNew_" + str(n) + ".json", "w"))
    n += 1


# In[ ]:


# Retrain network without kfold validation
mean = np.mean(Xtr, axis=0)
Xtr = np.subtract(Xtr,mean)
#Xts = np.subtract(Xts,mean)
std = np.std(Xtr, axis=0)
Xtr = np.divide(Xtr,std)
#Xts = np.divide(Xts,std)
class_weights = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2}
classifier = get_model(ytr, Xtr)
classifier.summary()
history = classifier.fit(Xtr, ytr,
                callbacks = [es],
                epochs = 40,
                batch_size = 32,
                validation_data = (Xts, yts),
                class_weight = class_weights)
yts_r = classifier.predict(Xts)
mcc_all.append(calc_mcc(yts, yts_r))
print('MCC: ', str(calc_mcc(yts, yts_r)))
output_txt.write(('MCC Final: ' + str(calc_mcc(yval, yval_rk))))
classifier.evaluate(Xts, yts)
classifier.save("./Models/model/noKFold_finalNew.h5")
json.dump(history.history, open("./Models/JSON/noKFold_finalNew.json", "w"))


# In[ ]:


avg, minimum, maximum, conf_int, inrange, outofrange = stats(mcc_tr, mcc_all)
output_txt.write(('Avg MCC: ' + str(avg)))
print(('Avg MCC: ' + str(avg)))
output_txt.write(('Min MCC: ' + str(minimum)))
print(('Min MCC: ' + str(minimum)))
output_txt.write(('Max MCC: ' + str(maximum)))
print(('Max MCC: ' + str(maximum)))
output_txt.write(('Confidence Interval: (' + str(conf_int[0]) + ', ' + str(conf_int[1]) + ')'))
print(('Confidence Interval: (' + str(conf_int[0]) + ', ' + str(conf_int[1]) + ')'))
output_txt.write((str(outofrange) + ' MCC values outside confidence interval'))
print((str(outofrange) + ' MCC values outside confidence interval'))
output_txt.close()

# In[ ]:


# In[ ]:
