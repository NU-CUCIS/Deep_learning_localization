import numpy as np
# import pandas as pd
import pickle
from keras import regularizers
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.core import Activation, Dense, Dropout
from keras.layers import Input, Dense, Flatten, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, UpSampling2D, ZeroPadding1D, GlobalAveragePooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.callbacks import Callback
import tensorflow as tf
# add this line because of update tensorflow
# tf.python.control_flow_ops = tf
from tensorflow.python.framework import ops
from scipy.io import loadmat
import random
import scipy.stats
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import matplotlib
matplotlib.use('Agg')
import pylab as plt
from keras.callbacks import LearningRateScheduler
from keras.regularizers import l2
import h5py
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle
import time


experiment_num = 'MKS1021_v2_3'
mve_len = 21
n_epoch=2000
learning_rate=0.001
batch_size = 21 * 3 * 4
local_cube_len = 11
l_cube = (local_cube_len-1)/2
r_cube = (local_cube_len+1)/2
L2 = 0.0001
scale = 10000.0


path = '/raid/zyz293/MKS_contrast50/'
print 'load original data'
data = scipy.io.loadmat(path+'MSdata.mat')
Mtrain = np.array(data['Mmatrix'])
Strain = np.array(data['Smatrix'])
indCal = np.array(data['indCal'])-1 # matlab indexing from 1, -1 to make it readable for python
indVal = np.array(data['indVal'])-1
train_data = np.reshape(Mtrain[indCal], (1500,21,21,21))
test_data = np.reshape(Mtrain[indVal], (1000,21,21,21))
del Mtrain
train_label = np.reshape(Strain[indCal], (1500,21,21,21))
test_label = np.reshape(Strain[indVal], (1000,21,21,21))
del Strain

# mean_strain_test = np.mean(train_label)
coordinate = []
for i in range(mve_len):
    for j in range(mve_len):
        for k in range(mve_len):
            temp = [i,j,k]
            coordinate.append(temp)
with open(path+'MVE_index_old.pkl', 'r') as f:
    MVE_index = np.array(pickle.load(f))
train_num = 1200
train_coordinate = MVE_index[:train_num]
validation_num = 300
validation_coordinate = MVE_index[train_num:]
validation_data = train_data[validation_coordinate]
validation_label = train_label[validation_coordinate]
train_data = train_data[train_coordinate]
train_label = train_label[train_coordinate]
######### normalize input data from (0,1) to (-0.5,0.5)
train_data = train_data - 0.5
validation_data = validation_data - 0.5
test_data = test_data - 0.5
train_label = scale * train_label
validation_label = scale * validation_label
######################
print 'train data: ', train_data.shape
print 'train label: ', train_label.shape
print 'validation data: ', validation_data.shape
print 'validation label: ', validation_label.shape
print 'test data: ', test_data.shape
print 'test label: ', test_label.shape

# create 2D CNN model
def build_model():
    print ('create model')
    model = Sequential()

    model.add(Convolution2D(128, 3, 3, init='glorot_normal', border_mode='valid', W_regularizer=l2(L2), dim_ordering='tf', input_shape=(local_cube_len,local_cube_len,local_cube_len)))
    model.add(Activation('relu'))
    # model.add(Convolution2D(64, 3, 3, init='glorot_normal', border_mode='same', W_regularizer=l2(L2)))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(64, 3, 3, init='glorot_normal', border_mode='same', W_regularizer=l2(L2)))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(64, 3, 3, init='glorot_normal', border_mode='same', W_regularizer=l2(L2)))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(64, 3, 3, init='glorot_normal', border_mode='same', W_regularizer=l2(L2)))
    # model.add(Activation('relu'))
    # model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(256, 3, 3, init='glorot_normal', border_mode='valid', W_regularizer=l2(L2)))
    model.add(Activation('relu'))
    # model.add(Convolution2D(128, 3, 3, init='glorot_normal', border_mode='same', W_regularizer=l2(L2)))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(128, 3, 3, init='glorot_normal', border_mode='same', W_regularizer=l2(L2)))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(128, 3, 3, init='glorot_normal', border_mode='same', W_regularizer=l2(L2)))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(128, 3, 3, init='glorot_normal', border_mode='same', W_regularizer=l2(L2)))
    # model.add(Activation('relu'))
    # model.add(AveragePooling2D(pool_size=(2, 2)))
    # model.add(Convolution2D(256, 3, 3, init='glorot_normal', border_mode='same', W_regularizer=l2(L2)))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(256, 3, 3, init='glorot_normal', border_mode='same', W_regularizer=l2(L2)))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(256, 3, 3, init='glorot_normal', border_mode='same', W_regularizer=l2(L2)))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(256, 3, 3, init='glorot_normal', border_mode='same', W_regularizer=l2(L2)))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(256, 3, 3, init='glorot_normal', border_mode='same', W_regularizer=l2(L2)))
    # model.add(Activation('relu'))
    # model.add(AveragePooling2D(pool_size=(2, 2)))
    # model.add(Convolution2D(512, 1, 1, init='glorot_normal', border_mode='same', W_regularizer=l2(L2)))
    # model.add(Activation('relu'))
    # model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dense(2048, init='glorot_normal', activation='relu', W_regularizer=l2(L2)))
    # model.add(Dropout(0.8))
    model.add(Dense(1024, init='glorot_normal', activation='relu', W_regularizer=l2(L2)))
    # model.add(Dropout(0.3))
    # model.add(Dense(512, init='glorot_normal', activation='relu', W_regularizer=l2(0.0001)))
    model.add(Dense(1, init='glorot_normal', W_regularizer=l2(L2)))
    return model

def generator(data, labels, shuffle, coordinate):
    while 1:
        mve_index = np.arange(len(data))
        if shuffle:
            np.random.shuffle(mve_index)
        # print (mve_index.shape)
        coordinate_index = np.arange(len(coordinate))
        if shuffle:
            np.random.shuffle(coordinate_index)
        # print (coordinate_index.shape)
        for i in range(mve_index.shape[0]):
            temp_mve = data[mve_index[i]]
            temp_label = labels[mve_index[i]]
            for j in range(coordinate_index.shape[0]/batch_size):
                temp = np.zeros((mve_len*3,mve_len*3,mve_len*3))
                for q in range(3):
                    for w in range(3):
                        for e in range(3):
                            temp[(q*mve_len):((q+1)*mve_len), (w*mve_len):((w+1)*mve_len), (e*mve_len):((e+1)*mve_len)] = temp_mve
                x = []
                y = []
                ind_list = [coordinate[k] for k in coordinate_index[j*batch_size:(j+1)*batch_size]]
                for cor in ind_list:
                    label_value = temp_label[cor[0],cor[1],cor[2]]
                    y.append(label_value)
                    data_temp = np.array(temp[(cor[0]+mve_len-l_cube):(cor[0]+mve_len+r_cube),(cor[1]+mve_len-l_cube):(cor[1]+mve_len+r_cube),(cor[2]+mve_len-l_cube):(cor[2]+mve_len+r_cube)])
                    x.append(data_temp)
                x = np.array(x)
                y = np.array(y)
                # print (x.shape)
                # print (y.shape)
                yield x, y

print ('-------------------------')
# train_step = len(train_data)*mve_len*mve_len*mve_len//batch_size
# validation_step = len(train_data)*mve_len*mve_len*mve_len//batch_size
train_step = len(train_data)*len(coordinate)//batch_size
validation_step = len(validation_data)*len(coordinate)//batch_size
training_generator = generator(train_data, train_label, True, coordinate)
validation_generator = generator(validation_data, validation_label, False, coordinate)
print 'train step: ', train_step
print 'validation step: ', validation_step
# compile model
print ('compile model')
model = build_model()
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
filepath = './weights/'+experiment_num+'_bestweights.hdf5'

print ('-------------------------')
print ('fit model')
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
checkpoint = ModelCheckpoint(filepath, monitor='val_mean_absolute_error', verbose=1, save_best_only=True)
history = model.fit_generator(generator=training_generator, steps_per_epoch=train_step, validation_data=validation_generator, validation_steps=validation_step, nb_epoch=n_epoch, callbacks=[early_stopping,checkpoint])

# summarize history for accuracy
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('MAE')
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('./result_plot/'+experiment_num+'_model_metrics.png')
plt.clf()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('./result_plot/'+experiment_num+'_model_loss.png')
plt.clf()
print ('--------------------------')
print ('predict data')


def test_generat(data, labels, shuffle, coordinate):
    while 1:
        mve_index = np.arange(len(data))
        if shuffle:
            np.random.shuffle(mve_index)
        # print (mve_index.shape)
        coordinate_index = np.arange(len(coordinate))
        if shuffle:
            np.random.shuffle(coordinate_index)
        # print (coordinate_index.shape)
        for i in range(mve_index.shape[0]):
            temp_mve = data[mve_index[i]]
            temp_label = labels[mve_index[i]]
            for j in range(coordinate_index.shape[0]/batch_size):
                temp = np.zeros((mve_len*3,mve_len*3,mve_len*3))
                for q in range(3):
                    for w in range(3):
                        for e in range(3):
                            temp[(q*mve_len):((q+1)*mve_len), (w*mve_len):((w+1)*mve_len), (e*mve_len):((e+1)*mve_len)] = temp_mve
                x = []
                y = []
                ind_list = [coordinate[k] for k in coordinate_index[j*batch_size:(j+1)*batch_size]]
                for cor in ind_list:
                    label_value = temp_label[cor[0],cor[1],cor[2]]
                    y.append(label_value)
                    final_test_label.append(label_value)
                    data_temp = np.array(temp[(cor[0]+mve_len-l_cube):(cor[0]+mve_len+r_cube),(cor[1]+mve_len-l_cube):(cor[1]+mve_len+r_cube),(cor[2]+mve_len-l_cube):(cor[2]+mve_len+r_cube)])
                    x.append(data_temp)
                x = np.array(x)
                y = np.array(y)
                # print (x.shape)
                # print (y.shape)
                yield x, y

del model 
del train_data
del train_label
del validation_data
del validation_label
test_coordinate = np.array(coordinate)
final_test_label = []
test_generator = test_generat(test_data, test_label, False, test_coordinate)
# train_step = len(train_data)*mve_len*mve_len*mve_len//batch_size
test_step = len(test_data)*len(test_coordinate)//batch_size
print 'test step: ', test_step
model = build_model()
model.load_weights(filepath)
final_pred_y = np.array(model.predict_generator(generator=test_generator, steps=test_step)) / scale
# the shape of prediction and labels should be the same, which is (N, 1)
print (final_pred_y.shape) 
# test_label = test_label.flatten()
mase = []
for i in range(len(test_data)):
    label_temp = final_test_label[i*len(test_coordinate):(i+1)*len(test_coordinate)]
    pred_temp = final_pred_y[i*len(test_coordinate):(i+1)*len(test_coordinate)]
    # print label_temp.shape
    # print pred_temp.shape
    mean_strain_test = np.mean(label_temp)
    mae = mean_absolute_error(label_temp, pred_temp)
    mase.append(mae / mean_strain_test)
sess = tf.Session()
print ('------------------------')
print (experiment_num)
print ('MASE: ', np.mean(np.array(mase)) * 100)




