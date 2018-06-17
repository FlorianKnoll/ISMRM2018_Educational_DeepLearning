#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DTI classification demo ISMRM 2018

Created in May 2018 for ISMRM educational "How to Jump-Start Your Deep Learning Research"
Educational course Deep Learning: Everything You Want to Know, Saturday, June 16th 2018
Joint Annual meeting of ISMRM and ESMRMB, Paris, France, June 16th to 21st

Created with PyTorch 0.4 and Python 3.6 using CUDA 8.0
Please see import section for module dependencies

florian.knoll@nyumc.org

"""

#%reset

#%% Import modules
import numpy as np
np.random.seed(123)  # for reproducibility
import pandas
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import os
torch.manual_seed(123)  # for reproducibility

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
#2: left genu of the corpus callosum
#3: left subcortical white matter of inferior frontal gyrus
data1 = pandas.read_csv("./data/dti/sampledata100206.csv", header=None).values 
data2 = pandas.read_csv("./data/dti/sampledata105620.csv", header=None).values 
data3 = pandas.read_csv("./data/dti/sampledata107725.csv", header=None).values 
data4 = pandas.read_csv("./data/dti/sampledata112314.csv", header=None).values 

data_cat = np.concatenate((data2,data3,data4),axis=0)

#%% Remove classes and slice position features
x_test = data1[:,4:9].astype(float)
y_test = data1[:,9]-1 # class labels are expected to start at 0
X = data_cat[:,4:9].astype(float)
Y = data_cat[:,9]-1 # class labels are expected to start at 0

#%% Normalize data
nSamples = np.size(Y)
nSamples_test = np.size(y_test)
nClasses = np.int(np.max(Y))+1
nFeatures = np.size(X,1)

for ii in range(0,nFeatures):
    feature_normalization = max(X[:,ii])
    X[:,ii] = X[:,ii]/feature_normalization
    x_test[:,ii] = x_test[:,ii]/feature_normalization
      
#%% Separate training and validation
setsize_train = np.ceil(nSamples*0.8).astype(int)
setsize_val = np.ceil(nSamples*0.2).astype(int)

#random permuation of data and classes
idx = np.random.permutation(nSamples)
idx_train = idx[0:setsize_train]
idx_val = idx[setsize_train:setsize_train+setsize_val]

x_train = X[idx_train,:]
y_train = Y[idx_train]

x_val = X[idx_val,:]
y_val = Y[idx_val]

#%%Generate torch variables 
x_train = torch.Tensor(x_train).float()
y_train = torch.Tensor(y_train).long()

x_val = torch.Tensor(x_val).float()
y_val = torch.Tensor(y_val).long()

x_test = torch.Tensor(x_test).float()
y_test = torch.Tensor(y_test).long()

#%% Check balancing of classes
#np.sum(y_train==0)
#np.sum(y_train==1)
#np.sum(y_train==2)
#
#np.sum(y_val==0)
#np.sum(y_val==1)
#np.sum(y_val==2)
#
#np.sum(y_test==0)
#np.sum(y_test==1)
#np.sum(y_test==2)
#
#np.sum(y_train==0)+np.sum(y_val==0)+np.sum(y_test==0)
#np.sum(y_train==1)+np.sum(y_val==1)+np.sum(y_test==1)
#np.sum(y_train==2)+np.sum(y_val==2)+np.sum(y_test==2)

#%% Define model
nElements = 100
nLayers = 3
model_name = 'dti_FC'
model = torch.nn.Sequential(
    torch.nn.Linear(nFeatures, nElements, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(nElements, nElements, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(nElements, nElements, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(nElements, nElements, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(nElements, nClasses, bias=True),
)
print(model)

#%%choose optimizer and loss function
training_epochs = 250
lr = 0.001
batch_size = 1024
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#%%Create minibatch data loading for training and validation
dataloader_train = data_utils.TensorDataset(x_train, y_train)
dataloader_train = data_utils.DataLoader(dataloader_train, batch_size=batch_size, shuffle=False,num_workers=4)

#%% Train model
loss_train = np.zeros(training_epochs)
acc_train = np.zeros(training_epochs)
loss_val = np.zeros(training_epochs)
acc_val = np.zeros(training_epochs)

for epoch in range(training_epochs):
    for local_batch, local_labels in dataloader_train:
        # feedforward - backpropagation
        optimizer.zero_grad()
        out = model(local_batch)
        loss = criterion(out, local_labels)
        loss.backward()
        optimizer.step()
        loss_train[epoch] = loss.item()
        # Training data accuracy
        [dummy, predicted] = torch.max(out.data, 1)
        acc_train[epoch] = (torch.sum(local_labels==predicted).numpy() / np.size(local_labels.numpy(),0))
 
        # Validation
        out_val = model(x_val)
        loss = criterion(out_val, y_val)
        loss_val[epoch] = loss.item()
        [dummy, predicted_val] = torch.max(out_val.data, 1)
        acc_val[epoch] = ( torch.sum(y_val==predicted_val).numpy() / setsize_val)
    
    print ('Epoch {}/{} train loss: {:.3}, train acc: {:.3}, val loss: {:.3}, val acc: {:.3}'.format(epoch+1, training_epochs, loss_train[epoch], acc_train[epoch], loss_val[epoch], acc_val[epoch]))
    
#%% Evaluate trained model
#Double check model on train data
out = model(x_train)
[dummy, predicted] = torch.max(out.data, 1)
acc_train_final = (torch.sum(y_train==predicted).numpy() / setsize_train)
print('Evaluation results train data: {:.2}'.format(acc_train_final))

#Double check model on validation data
out = model(x_val)
[dummy, predicted] = torch.max(out.data, 1)
acc_val_final = (torch.sum(y_val==predicted).numpy() / setsize_val)
print('Evaluation results validation data: {:.2}'.format(acc_val_final))

#Evaluate model on test data
out = model(x_test)
[dummy, predicted] = torch.max(out.data, 1)
acc_test_final = (torch.sum(y_test==predicted).numpy() / nSamples_test)
print('Evaluation results test data: {:.2}'.format(acc_test_final))

#%% Plot training overview
os.makedirs('./training_plots_pytorch')
plot_label = 'FC {} layers {} elements: train/val/test={:.2}/{:.2}/{:.2}'.format(nLayers,nElements,acc_train_final,acc_val_final,acc_test_final)
N=5
plt.figure(1)
plt.plot(np.convolve(acc_train, np.ones((N,))/N, mode='valid'))
plt.plot(np.convolve(acc_val, np.ones((N,))/N, mode='valid'))
plt.title(plot_label)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='lower right')
plt.ylim(0.5,0.9)
plt.show()
plt.savefig('./training_plots_pytorch/{}_{}layers_{}elements_epochs{}.png'.format(model_name,nLayers,nElements,training_epochs))