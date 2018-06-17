#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tensorflow linear regression demo ISMRM 2018

Created in May 2018 for ISMRM educational "How to Jump-Start Your Deep Learning Research"
Educational course Deep Learning: Everything You Want to Know, Saturday, June 16th 2018
Joint Annual meeting of ISMRM and ESMRMB, Paris, France, June 16th to 21st

Created with Tensorflow 1.2.1 using CUDA 8.0
Please see import section for module dependencies

florian.knoll@nyumc.org

"""
#%reset

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os

#%% training data
x_train = [1,2,3,4,5]
y_train = [3,5,7,9,11]

#%% Model parameters
k = tf.Variable([0.1], dtype=tf.float32)
d = tf.Variable([-0.1], dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
linear_model = (k * x + d)
y = tf.placeholder(tf.float32)

# sum of squares loss
loss = tf.reduce_sum(tf.square(linear_model - y))

#%% optimizer
lr = 0.005
optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

#%% train
training_epochs = 1000
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
loss_ii = np.zeros(training_epochs)
k_ii = np.zeros(training_epochs)
d_ii = np.zeros(training_epochs)

t = time.time()
for ii in range(training_epochs):
  [k_ii_temp, d_ii_temp, loss_ii[ii]] = sess.run([k, d, loss], {x:x_train, y:y_train})
  k_ii[ii] = k_ii_temp.item()
  d_ii[ii] = d_ii_temp.item()
  print('epoch: {}, k={:.3}, b={:.3}, loss={:.3}'.format(ii+1,k_ii[ii], d_ii[ii], loss_ii[ii]))
  sess.run(optimizer, {x:x_train, y:y_train})
  
elapsed = time.time() - t
print('Training time: {:.2} s'.format(elapsed))

#%% Plot training overview
os.makedirs('./training_plots_tensorflow')
plt.figure(1)
plt.plot(loss_ii)
plt.title('Linear regression loss')
plt.ylabel('SOS error (a.u.)')
plt.xlabel('epoch')
plt.show()
plt.savefig('./training_plots_tensorflow/linear_regression_epochs{}.png'.format(training_epochs))

#%% Plot training overview
plt.figure(2)
plt.plot(k_ii)
plt.plot(d_ii)
plt.title('Trained model parameters')
plt.ylabel('parameter values')
plt.xlabel('epoch')
plt.legend(['k', 'd'], loc='lower right')
plt.show()
plt.savefig('./training_plots_tensorflow/linear_regression_model_parameters_epochs{}.png'.format(training_epochs))