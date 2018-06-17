#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DTI classification demo ISMRM 2018

Created in May 2018 for ISMRM educational "How to Jump-Start Your Deep Learning Research"
Educational course Deep Learning: Everything You Want to Know, Saturday, June 16th 2018
Joint Annual meeting of ISMRM and ESMRMB, Paris, France, June 16th to 21st

Created with Tensorflow 1.2.1, Keras 2.0.6 and Python 3.6 using CUDA 8.0
Please see import section for module dependencies

florian.knoll@nyumc.org

"""

#%reset

#%% Import modules
import numpy as np
np.random.seed(123)  # for reproducibility
import pandas
import time
import os

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.9
#config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))

import keras as keras
from keras.models import Sequential
from keras.utils import plot_model
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

plt.close("all")

#%% Load dataset
# The first case is used as an independent test set. Cases 2-4 are used for training and validation
#
#Entries in the CVS file are
#1: sample
#2: row
#3: column
#4: slice
#5: T1 weighted anatomical image
#6: FA
#7: MD
#8: AD
#9: RD
#10: Label
#
#Classes are
#1: left thalamus
#3: left genu of the corpus callosum
#4: left subcortical white matter of inferior frontal gyrus
data1 = pandas.read_csv("./data/dti/sampledata100206.csv", header=None).values 
data2 = pandas.read_csv("./data/dti/sampledata105620.csv", header=None).values 
data3 = pandas.read_csv("./data/dti/sampledata107725.csv", header=None).values 
data4 = pandas.read_csv("./data/dti/sampledata112314.csv", header=None).values 

data_cat = np.concatenate((data2,data3,data4),axis=0)

#%% Remove classes and slice position features
x_test = data1[:,4:9].astype(float)
y_test = data1[:,9]
X = data_cat[:,4:9].astype(float)
Y = data_cat[:,9]
#x_test = data1[:,[5,6]].astype(float)
#y_test = data1[:,9]
#X = data_cat[:,[5,6]].astype(float)
#Y = data_cat[:,9]
#data1[1,:]
#X[1,:]
#x_test[1,:]
#Y[1]

#%% Normalize data
nSamples = np.size(Y)
nClasses = np.int(np.max(Y))
nFeatures = np.size(X,1)

for ii in range(0,nFeatures):
    feature_normalization = max(X[:,ii])
    X[:,ii] = X[:,ii]/feature_normalization
    x_test[:,ii] = x_test[:,ii]/feature_normalization

#%% Separate training and validation
setsize_train = np.ceil(nSamples*0.8).astype(int)
setsize_val = np.ceil(nSamples*0.2).astype(int)

#encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
encoded_y_test = encoder.transform(y_test)

# convert to one hot encoded
Y = np_utils.to_categorical(encoded_Y)
y_test = np_utils.to_categorical(encoded_y_test)

#random permuation of data and classes
idx = np.random.permutation(nSamples)
idx_train = idx[0:setsize_train]
idx_val = idx[setsize_train:setsize_train+setsize_val]

x_train = X[idx_train,:]
y_train = Y[idx_train,:]

x_val = X[idx_val,:]
y_val = Y[idx_val,:]

#%% Define model
nElements = 100
nLayers = 3
model = Sequential()
model.add(Dense(nElements, input_dim=nFeatures, activation='relu',name="input_layer"))
model.add(Dense(nElements, activation='relu',name="hidden_layer_01"))
model.add(Dense(nElements, activation='relu',name="hidden_layer_02"))
model.add(Dense(nElements, activation='relu',name="hidden_layer_03"))
model.add(Dense(nClasses, activation='softmax', name="output_softmax"))
model_name = 'dti_FC_{}layers_{}elements'.format(nLayers,nElements)

# Compile model
# default adam learning rate: 0.001 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#%% Plot model information
os.makedirs('./models')
model.summary()
with open('./models/{}.txt'.format(model_name),'w') as fh:
    model.summary(print_fn=lambda x: fh.write(x + '\n'))
    
#plot_model(model, to_file='./models/{}.png'.format(model_name))

#%% Set up tensorboard
tensorboard_graphdir = './graph_dti/{0}_{1}'.format(time.strftime('%Y-%m-%d_%H-%M-%S'),model_name)
os.makedirs(tensorboard_graphdir)
tbCallBack = keras.callbacks.TensorBoard(log_dir=tensorboard_graphdir, histogram_freq=0, write_graph=True, write_images=True)

#%% Train model
training_epochs = 250
batch_size=1024

print("Train",model_name,"dti classification")
print("Training data points: {}".format(np.size(y_train,0)))
print("Validation data points: {}".format(np.size(y_val,0)))
print("Test data points: {}".format(np.size(y_test,0)))

training_history=model.fit(x_train, y_train, batch_size=batch_size,  validation_data=(x_val,y_val), epochs=training_epochs, verbose=1,callbacks=[tbCallBack])
 
#%% Evaluate trained model
#Double check model on train data
print('=========================================')
print("Evaluate on {} train images".format(np.size(y_train,0)))
score_train = model.evaluate(x_train, y_train, verbose=1)
print("Evaluation results train data: loss: {:.3} acc: {:.4}".format(score_train[0], score_train[1]))

#Double check model on validation data
print('=========================================')
print("Evaluate on {} train images".format(np.size(y_train,0)))
score_val = model.evaluate(x_val, y_val, verbose=1)
print("Evaluation results validation data: loss: {:.3} acc: {:.4}".format(score_val[0], score_val[1]))

#Evaluate model on test data
print('=========================================')
print("Evaluate on {} test images".format(np.size(y_test,0)))
score_test = model.evaluate(x_test, y_test, verbose=1)
print("Evaluation results test data: loss: {:.3} acc: {:.4}".format(score_test[0], score_test[1]))

#%% Plot training overview
os.makedirs('./training_plots_tensorflow')
plot_label = 'FC {} layers {} elements: train/val/test={:.2}/{:.2}/{:.2}'.format(nLayers,nElements,score_train[1],score_val[1],score_test[1])
N=5
plt.figure(1)
plt.plot(np.convolve(training_history.history['acc'], np.ones((N,))/N, mode='valid'))
plt.plot(np.convolve(training_history.history['val_acc'], np.ones((N,))/N, mode='valid'))
plt.title(plot_label)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='lower right')
plt.ylim(0.5,0.9)
plt.show()
plt.savefig('./training_plots_tensorflow/{}_epochs{}.png'.format(model_name,training_epochs))