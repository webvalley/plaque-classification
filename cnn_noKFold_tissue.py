#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os
import pickle
import json
import cv2
import sys
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

output_txt = open("output5tissue.txt","w")
output_txt.write('\nLR: 1e-5\n')
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

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

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
    classifier.add(Dense(units = 2, activation = 'softmax'))
    adam = keras.optimizers.Adam(lr = 1e-5)
    classifier.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy', f1_m, precision_m, recall_m])
    return classifier



# def get_model(ytr, Xtr):
#     classifier = Sequential()
#     classifier.add(Conv2D(16, (5, 5), input_shape = (64, 64, 1), activation = 'relu'))
#     classifier.add(Dropout(0.4))
#     classifier.add(MaxPooling2D(pool_size = (4, 4)))
#     classifier.add(Conv2D(32, (4, 4), activation = 'relu'))
#     classifier.add(Dropout(0.2))
#     classifier.add(MaxPooling2D(pool_size = (4, 4)))
#     classifier.add(Flatten())
#     classifier.add(Dense(units = 128, activation = 'relu'))
#     classifier.add(Dropout(0.3))
#     classifier.add(Dense(units = 32, activation = 'relu'))
#     classifier.add(Dense(units = 2, activation = 'softmax'))
#     adam = keras.optimizers.Adam(lr = 1e-6)
#     classifier.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy', f1_m, precision_m, recall_m])
#     return classifier



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
'''
count0 = 0
dir = "./Data/Patches/NoPlaque"
for filename in os.listdir(dir):
    img = np.array(cv2.imread(os.path.join(dir, filename)))[:,:,0:1]
    Xtemp.append(img)
    ytemp.append([1 , 0])
    
while(len(index) != 2510):
    r = random.randint(1,2510)
    if r not in index: 
        index.append(r)
        
for i in index:
    X.append(Xtemp[i])
    y.append(ytemp[i])
count0 = len(X)
'''

count0 = 0
dir = "./Data/Patches/Plaque/Aug/cal"
for filename in os.listdir(dir):
    img = np.array(cv2.imread(os.path.join(dir, filename)))[:,:,0:1]
    X.append(img)
    y.append([0, 1])
    count0 += 1


count1 = 0
dir = "./Data/Patches/Plaque/Aug/fibrous"
for filename in os.listdir(dir):
    img = np.array(cv2.imread(os.path.join(dir, filename)))[:,:,0:1]
    X.append(img)
    y.append([1, 0])  
    count1 += 1

X = np.array(X)
y = np.array(y)

weight_for_0 = 1. / count0
weight_for_1 = 1. / count1
#weight_for_2 = 1. / count2

print("count #0: ", count0, " | count #1: ", count1, "\n")
print("weight for 0: ", weight_for_0, " | weight for 1: ", weight_for_1)
output_txt.write(("count #0: "+ str(count0) + " | count #1: "+ str(count1)+ "\n"))
output_txt.write(("weight for 0: "+ str(weight_for_0)+ " | weight for 1: "+ str(weight_for_1)+ "\n"))
                 
Xtr, Xts, ytr, yts = train_test_split(np.copy(X), np.copy(y), test_size=0.25, stratify=y, random_state=42)

#ytr_categorical = keras.utils.to_categorical(ytr, num_classes = 3)

ytr_onedim = np.argmax(ytr, axis=1)
rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=42)
mcc_all = []
mcc_tr = []
n = 0

for train_index, val_index in rskf.split(Xtr, ytr_onedim):
    #print("TRAIN:", train_index, "VAL:", val_index)
    Xtrk, Xval = Xtr[train_index], Xtr[val_index]
    ytrk, yval = ytr[train_index], ytr[val_index]
    
    mean_k = np.mean(Xtrk)
    std_k = np.std(Xtrk)
    print(("mean: " + str(mean_k) + " | std: " + str(std_k)))
    output_txt.write(("mean:" + str(mean_k) + "\nstd:" + str(std_k) + "\n"))
    
    Xtrk = (Xtrk - mean_k) / std_k
    Xval = (Xval - mean_k) / std_k
    
    print("Xtrk mean:", Xtrk.mean(), "Xtrk std:", Xtrk.std())
    print("Xval mean:", Xval.mean(), "Xval std:", Xval.std())
    
    print("#ytrk:", ytrk.sum(axis=0), "#yval:", yval.sum(axis=0))
    print("#ytrk:", ytrk.sum(axis=0)/len(ytrk), "#yval:", yval.sum(axis=0) / len(yval))

    classifier = get_model(ytrk, Xtrk)
    classifier.summary()

    print("Run: ", n)
    #es = EarlyStopping(patience=5, restore_best_weights = True)
    class_weights = {0: weight_for_0, 1: weight_for_1} #2: weight_for_2}
    history = classifier.fit(Xtrk, ytrk,
                    #callbacks = [es],
                    epochs = 40,
                    batch_size = 32,
                    validation_data = (Xval, yval),
                    class_weight = class_weights)

    yval_rk = classifier.predict(Xval)
    mcc_tr.append(calc_mcc(yval, yval_rk))
    print('MCC ', str(n), ': ', str(calc_mcc(yval, yval_rk)))
    output_txt.write(('\nMCC for fold ' + str(n) + ': ' + str(calc_mcc(yval, yval_rk)) + '\n'))
    classifier.evaluate(Xval, yval)
    classifier.save("./Models/model/tests/KFoldTissue_lr-5_" + str(n) + ".h5")
    json.dump(history.history, open("./Models/JSON/tests/KFoldTissue_lr-5_" + str(n) + ".json", "w"))
    n += 1
    
    #print("DOING ONLY ONE SEED")
    #break

#sys.exit(0)

# In[ ]:

# Retrain network without kfold validation

mean = np.mean(Xtr)
std = np.std(Xtr)

Xtr = (Xtr - mean) / std
Xts = (Xts - mean) / std
print("mean:", str(mean), "std:", str(std))
output_txt.write(("mean:" + str(mean) + '\n' + "std:" + str(std) + '\n'))
#plaque vs noplaque mean: 43.35145848896082 std: 34.45658024674099

class_weights = {0: weight_for_0, 1: weight_for_1} #, 2: weight_for_2}
classifier = get_model(ytr, Xtr)
classifier.summary()
history = classifier.fit(Xtr, ytr,
                #callbacks = [es],
                epochs = 40,
                batch_size = 32,
                validation_data = (Xts, yts),
                class_weight = class_weights)
yts_r = classifier.predict(Xts)
mcc_all.append(calc_mcc(yts, yts_r))
print('MCC: ', str(calc_mcc(yts, yts_r)))
output_txt.write(('\nMCC Final: ' + str(calc_mcc(yts, yts_r)) + '\n'))
classifier.evaluate(Xts, yts)
classifier.save("./Models/model/tests/noKFoldTissue_lr-5.h5")
json.dump(history.history, open("./Models/JSON/tests/noKFoldTissue_lr-5.json", "w"))

# In[ ]:

avg, minimum, maximum, conf_int, inrange, outofrange = stats(mcc_tr, mcc_all)
output_txt.write(('\nAvg MCC: ' + str(avg) + '\n'))
print(('Avg MCC: ' + str(avg)))
output_txt.write(('Min MCC: ' + str(minimum) + '\n'))
print(('Min MCC: ' + str(minimum)))
output_txt.write(('Max MCC: ' + str(maximum) + '\n'))
print(('Max MCC: ' + str(maximum)))
output_txt.write(('Confidence Interval: (' + str(conf_int[0]) + ', ' + str(conf_int[1]) + ')\n'))
print(('Confidence Interval: (' + str(conf_int[0]) + ', ' + str(conf_int[1]) + ')'))
output_txt.write((str(outofrange) + ' MCC values outside confidence interval\n'))
print((str(outofrange) + ' MCC values outside confidence interval'))
output_txt.close()

# In[ ]:
