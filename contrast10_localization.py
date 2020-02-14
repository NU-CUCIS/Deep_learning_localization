import numpy as np
import pickle
from keras.models import Model, Sequential
from keras.layers.core import Activation, Dense
from keras.layers import Input, Dense, Flatten, Convolution2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.regularizers import l2
import tensorflow as tf
from tensorflow.python.framework import ops
import h5py
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')

# parametter setting
mve_len = 21 # length of 3D microstructure
n_epoch= 2 # number of training epoch
batch_size = 2 # batch size
local_cube_len = 11 # size of local cube centered at focal voxel
l_cube = (local_cube_len-1)/2
r_cube = (local_cube_len+1)/2
L2 = 0.0001 # penalty of L2 regularization
scale = 10000.0 # rescale output

# load data
with open('./sample_data.pkl', 'rb') as f:
    data = pickle.load(f)
train_data = data['data']
train_label = data['label']
# normalize input data from (0,1) to (-0.5,0.5)
train_data = train_data - 0.5
train_label = scale * train_label

coordinate = []
for i in range(mve_len):
    for j in range(mve_len):
        for k in range(mve_len):
            temp = [i,j,k]
            coordinate.append(temp)

# create 2D CNN model
def build_model():
    print ('create model')
    model = Sequential()

    model.add(Convolution2D(128, 3, 3, init='glorot_normal', border_mode='valid', W_regularizer=l2(L2), dim_ordering='tf', input_shape=(local_cube_len,local_cube_len,local_cube_len)))
    model.add(Activation('relu'))
    model.add(Convolution2D(256, 3, 3, init='glorot_normal', border_mode='valid', W_regularizer=l2(L2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(2048, init='glorot_normal', activation='relu', W_regularizer=l2(L2)))
    model.add(Dense(1024, init='glorot_normal', activation='relu', W_regularizer=l2(L2)))
    model.add(Dense(1, init='glorot_normal', W_regularizer=l2(L2)))
    return model

def generator(data, labels, shuffle, coordinate):
    mve_index = np.arange(len(data))
    if shuffle:
        np.random.shuffle(mve_index)
    coordinate_index = np.arange(len(coordinate))
    if shuffle:
        np.random.shuffle(coordinate_index)
    X = []
    Y = []
    for i in range(mve_index.shape[0]):
        temp_mve = data[mve_index[i]]
        temp_label = labels[mve_index[i]]
        # for j in range(coordinate_index.shape[0]/batch_size):
        temp = np.zeros((mve_len*3,mve_len*3,mve_len*3))
        for q in range(3):
            for w in range(3):
                for e in range(3):
                    temp[(q*mve_len):((q+1)*mve_len), (w*mve_len):((w+1)*mve_len), (e*mve_len):((e+1)*mve_len)] = temp_mve
        # ind_list = [coordinate[k] for k in coordinate_index[j*batch_size:(j+1)*batch_size]]
        # for cor in ind_list:
        for cor in coordinate:
            label_value = temp_label[cor[0],cor[1],cor[2]]
            Y.append(label_value)
            cor1l, cor1r = int(cor[0]+mve_len-l_cube), int(cor[0]+mve_len+r_cube)
            cor2l, cor2r = int(cor[1]+mve_len-l_cube), int(cor[1]+mve_len+r_cube)
            cor3l, cor3r = int(cor[2]+mve_len-l_cube), int(cor[2]+mve_len+r_cube)
            data_temp = temp[cor1l:cor1r, cor2l:cor2r, cor3l:cor3r]
            X.append(data_temp)
    return np.asarray(X), np.asarray(Y)

print ('-------------------------')
train_data, train_label = generator(train_data, train_label, True, coordinate)
print (train_data.shape, train_label.shape)
print ('compile model')
model = build_model()
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
filepath = './my_model.h5'

print ('-------------------------')
print ('fit model')
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
checkpoint = ModelCheckpoint(filepath, monitor='val_mean_absolute_error', verbose=1, save_best_only=True)
history = model.fit(train_data, train_label, batch_size = batch_size, validation_split=0.2, nb_epoch=n_epoch, callbacks=[early_stopping,checkpoint])

