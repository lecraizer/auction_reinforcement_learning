#!/usr/bin/env python
# -*- coding: utf-8 -*-

####### -------- Import packages -------- ####### 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import sys
import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from pygame import mixer
from numpy.random import choice
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.layers import Dense, LSTM, Input, concatenate


####### -------- Initialize parameters -------- ####### 

n_epochs = 5001
BS = 8 # size of training batch per epoch
if len(sys.argv) > 1:
    n_epochs = int(sys.argv[1])
if len(sys.argv) > 2:
    BS = int(sys.argv[2])

print('Total number of epochs:', n_epochs)
print('Batch size:', BS)


####### -------- Auxiliar functions -------- ####### 

def create_batch(batch_size=8):
    '''
    Create new data input for the model in which all players are 
    fixed and the other learns from the other ones' strategies
    '''
    X = []
    y = []

    X1 = []
    X2 = []

    for k in range(batch_size):
        x1 = random.random()
        x2 = random.random()
        # X.append( (x1, x2) )
        X1.append( x1 )
        X2.append( x2 )
        R = x2 - (x1-x2)**2
        y.append(R)

    # return np.array(X), np.array(y)
    return np.array(X1), np.array(X2), np.array(y)


def create_model(prec=10, loss='mse'):
    ''' 
    Initialize and create model
    '''
    first_input = Input(shape=(1, ))
    first_dense = Dense(1, use_bias=False)(first_input)

    second_input = Input(shape=(1, ))
    second_dense = Dense(1, use_bias=False)(second_input)

    merged = concatenate([first_dense, second_dense])
    x3 = Dense(20, activation='tanh', use_bias=False)(merged)
    x3 = Dense(100, activation='tanh', use_bias=False)(x3)
    x3 = Dense(400, activation='tanh', use_bias=False)(x3)
    x3 = Dense(100, activation='tanh', use_bias=False)(x3)
    x3 = Dense(20, activation='tanh', use_bias=False)(x3)

    # Final layer
    result = Dense(1, activation='linear', use_bias=False)(x3)

    # model = Model(inputs=[first_input, second_input], outputs=merged)
    model = Model(inputs=[first_input, second_input], outputs=result)

    model.compile(optimizer=opt, loss=loss)
    plot_model(model, to_file='stuff/model_plot.png', show_shapes=True, show_layer_names=False)
    print(model.summary())
    return model


def train(epochs, batch_size=BS, save_interval=1000):
    ''' Main function, used for training
    '''
    for epoch in range(epochs):
        # X_batch, y_batch = create_batch(batch_size=batch_size)
        X_batch1, X_batch2, y_batch = create_batch(batch_size=batch_size)
        # loss = model.train_on_batch(x=X_batch, y=y_batch)
        loss = model.train_on_batch(x=[X_batch1, X_batch2], y=y_batch)
        
        if epoch % save_interval == 0:
            print ("Epoch: %d | Loss: %f" % (epoch, loss))


####### -------- Initialize hiperparameters -------- ####### 

LR = 0.0001 # learning rate
opt = tf.keras.optimizers.Adam(learning_rate=LR) # optimizer


####### -------- Initialize and train model -------- ####### 

loss_func = 'mse'
print('Loss function:', loss_func)
print('Initializing model...')
model = create_model(loss_func)

print('Begin training!\n')
train(n_epochs, batch_size=BS)


####### -------- Create new model with frozen layers -------- ####### 

print('\nFirst model:\n')
for layer in model.layers:
    print(layer.name, layer.trainable)

new_model = create_model(loss_func)
weights = model.get_weights()
new_model.set_weights(weights)

for layer in new_model.layers:
    if layer.name != 'dense_9':
        layer.trainable = False

print('\nSecond model:\n')
for layer in new_model.layers:
    print(layer.name, layer.trainable)

print('\n\n')
print('Layer name:', new_model.layers[2].name)

w = new_model.layers[2].get_weights()[0][0][0]
print('Weights:', w)

### input = [1, 0.8]
### w treinável
### L = (100 - ŷ)

####### -------- Test results -------- ####### 

'''
print('\nTesting results!\n')

x1 = np.linspace(0, 1, 10)
x2 = np.linspace(0, 1, 10)

total_sum = 0.0
k = 0 
for a in x1:
    for b in x2:
        k += 1
        y_true = (a - b)**2 - b
        y_pred = new_model.predict([np.array([a]), np.array([b])])
        total_sum += (y_true - y_pred) 
        print(y_true - y_pred)
'''
