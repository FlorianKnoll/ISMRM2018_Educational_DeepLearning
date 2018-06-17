#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train CNN for classification of compressed sensing 4 times accelerated vs fully sampled reference reconstructions
Didactic example to demonstrate model over- and underfitting

Created in May 2018 for ISMRM educational "How to Jump-Start Your Deep Learning Research"
Educational course Deep Learning: Everything You Want to Know, Saturday, June 16th 2018
Joint Annual meeting of ISMRM and ESMRMB, Paris, France, June 16th to 21st

Created with Tensorflow 1.2.1, Keras 2.0.6 and Python 3.6 using CUDA 8.0
Please see import section for module dependencies

florian.knoll@nyumc.org
"""
 
#%reset

#%% import packages
from sklearn.preprocessing import LabelEncoder
from imutils import paths
import cv2
import time

import os,os.path

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
set_session(tf.Session(config=config))

import keras as keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D
from keras.utils import plot_model

import numpy as np
np.random.seed(123)  # for reproducibility
from matplotlib import pyplot as plt

from rgb2gray import rgb2gray
 
plt.close("all")

#%% Paths and training parameters
path_base = "./data/recon_classification/";

recon1 = "ref"
recon2 = "tgv"

training_epochs = 1000

print("paths for training and test image data")
path_base_training_recon1 = "{}{}{}".format(path_base,"train/",recon1,"/")
path_base_training_recon2 = "{}{}{}".format(path_base,"train/",recon2,"/")
path_base_testing_recon1 = "{}{}{}".format(path_base,"test/",recon1,"/")
path_base_testing_recon2 = "{}{}{}".format(path_base,"test/",recon2,"/")

trainingPaths = list(paths.list_images(path_base_training_recon1)) + list(paths.list_images(path_base_training_recon2))
testingPaths = list(paths.list_images(path_base_testing_recon1)) + list(paths.list_images(path_base_testing_recon2))

nTrainImages = len(trainingPaths)
nTestImages = len(testingPaths)

#%% Load data
# initialize data matrix and labels list
trainData   = []
trainLabels = []
testData    = []
testLabels  = []

# Training data
print("loading {} train images".format(nTrainImages))
for (i, imagePath) in enumerate(trainingPaths):
    image = cv2.imread(imagePath)
    image = rgb2gray(image)
    [nR,nC] = image.shape
    image = image.reshape(nR, nC, 1)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]
    trainData.append(image)
    trainLabels.append(label)

    if i > 0 and i % 1000 == 0:
        print("train images processed {}/{}".format(i, nTrainImages))

# Random shuffling: Do this manually here
print("Shuffling training data")
idx_train_rand = np.random.permutation(nTrainImages)
trainData_rand   = []
trainLabels_rand = []
for i in range(0, nTrainImages):
    # print("i: {}".format(i))
    trainData_rand.append(trainData[idx_train_rand[i]])
    trainLabels_rand.append(trainLabels[idx_train_rand[i]])

trainData = trainData_rand
del trainData_rand
trainLabels = trainLabels_rand
del trainLabels_rand

# In case we want to plot one of our images
#plt.figure(1)
#plt.imshow(trainData[1][:,:,0],cmap="gray")
#plt.axis('off')
#plt.title(trainLabels[1])

# Test data
print("loading {} test images".format(nTestImages))
for (i, imagePath) in enumerate(testingPaths):
    image = cv2.imread(imagePath)
    image = rgb2gray(image)
    [nR,nC] = image.shape
    image = image.reshape(nR, nC, 1)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]
    testData.append(image)
    testLabels.append(label)

    if i > 0 and i % 1000 == 0:
        print("test images processed {}/{}".format(i, nTestImages))
        
# In case we want to plot one of our test images
#plt.figure(2)
#plt.imshow(testData[1][:,:,0],cmap="gray")
#plt.axis('off')
#plt.title(testLabels[1])

#%% encode the labels, converting them from strings to integers
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)
testLabels = le.fit_transform(testLabels)

# scale the input image pixels to the range [0,1]
trainData = np.array(trainData) / 255.0
testData = np.array(testData) / 255.0

#%% Define model architecture
model = Sequential()
model.add(Convolution2D(4, (3, 3), activation='relu', padding='same', input_shape=(nR,nC,1)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(4, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(4, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(4, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Global Average Pooling
#model_name = 'CNN1layers_global_avg';
#model_name = 'CNN4layers_global_avg';
#model.add(GlobalAveragePooling2D())

# Fully connected and dense layer
model_name = 'CNN4layers_FC';
model.add(Flatten())
model.add(Dense(16, activation='relu'))

# Output layer
model.add(Dense(1, activation='sigmoid'))
 
#%% Plot model information
os.makedirs('./models')
model.summary()
with open('./models/{}.txt'.format(model_name),'w') as fh:
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

#plot_model(model, to_file='./models/{}.png'.format(model_name))

#%% Set up tensorboard
tensorboard_graphdir = './graph_recon/{0}_{1}_{2}_{3}'.format(time.strftime('%Y-%m-%d_%H-%M-%S'),recon1,recon2,model_name)
os.makedirs(tensorboard_graphdir)
tbCallBack = keras.callbacks.TensorBoard(log_dir=tensorboard_graphdir, histogram_freq=0, write_graph=True, write_images=True)

#%% Train and test
# default adam learning rate: 0.001 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Train using {} training images".format(nTrainImages))
training_history=model.fit(trainData, trainLabels, batch_size=10,  validation_data=(testData,testLabels), epochs=training_epochs, verbose=1,callbacks=[tbCallBack])
 
# Double check model on train data
print('=========================================')
print("Evaluate on {} train images".format(nTrainImages))
score_train = model.evaluate(trainData, trainLabels, verbose=1)
print("Evaluation results train data: loss: {:.3} acc: {:.4}".format(score_train[0], score_train[1]))

# Evaluate model on test data
print('=========================================')
print("Evaluate on {} test images".format(nTestImages))
score_val = model.evaluate(testData, testLabels, verbose=1)
print("Evaluation results test data: loss: {:.3} acc: {:.4}".format(score_val[0], score_val[1]))

#%% Evaluate predicted labels
predictedLabels = model.predict(testData,batch_size=8, verbose=1)
predictedLabels_round = np.round(predictedLabels)
predictedLabels_round = np.squeeze(np.transpose(predictedLabels_round.astype(int)))
predictedLabels_name = np.round(predictedLabels)
predictedLabels_name = predictedLabels_name.astype(int)
predictedLabels_name = le.inverse_transform(predictedLabels_name)  
predictedLabels_name = np.squeeze(predictedLabels_name)
testLabels_name = le.inverse_transform(testLabels)  
correctClassifications = np.transpose(testLabels==predictedLabels_round)

#%% Plot convergence
os.makedirs('./training_plots_tensorflow')
if model_name == "CNN1layers_global_avg":
    # plot_label = "1 conv layer global average"
    plot_label_train = '1 conv layer global average: train={:.2}'.format(score_train[1])
    plot_label_train_val = '1 conv layer global average: train/val={:.2}/{:.2}'.format(score_train[1],score_val[1])
elif model_name == "CNN4layers_global_avg":
    # plot_label = "4 conv layers global average"
    plot_label_train = '4 conv layers global average: train={:.2}'.format(score_train[1])
    plot_label_train_val = '4 conv layers global average: train/val={:.2}/{:.2}'.format(score_train[1],score_val[1])
elif model_name == "CNN4layers_FC":
     # plot_label = "4 conv layers fully connected"
     plot_label_train = '4 conv layers fully connected: train={:.2}'.format(score_train[1])
     plot_label_train_val = '4 conv layers fully connected: train/val={:.2}/{:.2}'.format(score_train[1],score_val[1])
else:
     print('This model does not exist')

N=50
plt.figure(1)
plt.plot(np.convolve(training_history.history['acc'], np.ones((N,))/N, mode='valid'))
plt.title(plot_label_train)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training'], loc='lower right')
plt.ylim(0.4,1.1)
plt.show()
plt.savefig('./training_plots_tensorflow/{}_{}_{}_{}_train.png'.format(model_name,training_epochs,recon1,recon2))

plt.figure(2)
plt.plot(np.convolve(training_history.history['acc'], np.ones((N,))/N, mode='valid'))
plt.plot(np.convolve(training_history.history['val_acc'], np.ones((N,))/N, mode='valid'))
plt.title(plot_label_train_val)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='lower right')
plt.ylim(0.4,1.1)
plt.show()
plt.savefig('./training_plots_tensorflow/{}_{}_{}_{}_train_val.png'.format(model_name,training_epochs,recon1,recon2))

