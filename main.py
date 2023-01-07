import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import sys
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy.random import choice
import keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import os

from pygame import mixer # Load the required library



####### -------- Initialize parameters -------- ####### 

grid_division = 5
n_epochs = 5001
BS = 8 # size of training batch per epoch
if len(sys.argv) > 1:
    grid_division = int(sys.argv[1]) # division of bids in the interval (0,1)
if len(sys.argv) > 2:
    n_epochs = int(sys.argv[2])
if len(sys.argv) > 3:
    BS = int(sys.argv[3])

print('\nGrid division:', grid_division)
print('Total number of epochs:', n_epochs)
print('Batch size:', BS)
N = 2 # number of players

####### -------- Auxiliar functions -------- ####### 

def create_batch(fixed_b, batch_size=8, prec=5):
    '''
    Create new data input for the model in which all players are 
    fixed and the other learns from the other ones' strategies
    '''
    # Create empty arrays
    X = []
    y = []

    for k in range(batch_size):
        
        # Learnable player
        v = random.random()
        X.append(v)

        payoffs = []
        for x in range(prec):
            b = x/prec
            # Game on!
            if b > fixed_b: # player wins the auction
                R = v - b
            elif b == fixed_b: # draw
                R = (v - b) / 2
            else: # player loses the auction
                R = 0
            
            # Result of trainable player
            payoffs.append(R)

        y.append(payoffs)

    return np.array(X), np.array(y)


def initialize_players(N=2, prec=10, loss='mse'):
    ''' 
    Initialize players with the same model architecture
    '''
    #@title
    players = []
    for i in range(N):
        model = keras.Sequential([ 
            keras.layers.Flatten(input_shape=(1,)),
            keras.layers.Dense(20, activation=tf.nn.tanh),
            keras.layers.Dense(100, activation=tf.nn.tanh),
            keras.layers.Dense(400, activation=tf.nn.tanh), 
            keras.layers.Dense(100, activation=tf.nn.tanh),
            keras.layers.Dense(prec),
        ])

        # model.compile(loss='categorical_crossentropy', optimizer=opt)
        model.compile(loss=loss, optimizer=opt, run_eagerly=True)
        players.append(model)
    return players

# weights = np.array([0.4, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.1, 0.07, 0.03])
# weights = np.array([1/grid_division]*grid_division)
weights = [0.01, 0.02, 0.05, 0.25, 0.3, 0.25, 0.08, 0.02, 0.01, 0.01]
# weights = [0.005, 0.01, 0.01, 0.01, 0.02, 0.04, 0.06, 0.23, 0.27, 0.23, 0.05, 0.015, 0.015, 0.015, 0.005, 0.005, 0.0025, 0.0025, 0.0025, 0.0025]

def custom_mse_loss(y_true, y_pred):
    idxs = K.get_value(K.argmax(y_true, axis=1))
    '''
    W = [[0.0]*y_true.shape[1] for i in range(y_true.shape[0])]
    for j in range(len(idxs)): # size of batch
        if idxs[j] == 0:
            W[j] = [1.0]*10
        else:
            W[j][idxs[j]] += 0.2
            if idxs[j] > 0:
                W[j][idxs[j]-1] += 0.1
            if idxs[j] < y_true.shape[1]-1:
                W[j][idxs[j]+1] += 0.1
            leftover = 1 - sum(W[j])
            num_zeros = W[j].count(0)
            factor = leftover/num_zeros
            for k in range(len(W[j])):
                if W[j][k] == 0.0:
                    W[j][k] += factor
                W[j][k] *= 10
    # return K.sum(W * K.square(y_pred-y_true))
    '''
    return K.sum(K.square(y_pred-y_true))
    

# def custom_mse_loss(y_true, y_pred):
#     idx = np.argmax(np.array(y_true))
#     print(idx)
#     W = [0.2/grid_division]*grid_division
#     W[idx] += 0.8
#     return W * K.square(y_pred-y_true)
    # return K.mean(W * K.square(y_pred-y_true))


def train(epochs, batch_size=BS, prec=grid_division, save_interval=1000):
    ''' Main function, used for training
    '''
    # First iteraction
    X_batch, y_batch = create_batch(random.random(), batch_size=batch_size, prec=prec)
    loss2 = player2.train_on_batch(x=X_batch, y=y_batch)

    for epoch in range(epochs):

        # Player 2 fixed, train player 1
        v2 = random.random() # value from fixed player
        b2 = np.argmax(player2.predict([v2])) / prec # bid from fixed player
        X_batch, y_batch = create_batch(b2, batch_size=batch_size, prec=prec)
        loss1 = player1.train_on_batch(x=X_batch, y=y_batch)

        # Player 1 fixed, train player 2
        v1 = random.random()
        b1 = np.argmax(player1.predict([v1])) / prec
        X_batch, y_batch = create_batch(b1, batch_size=batch_size, prec=prec)
        loss2 = player2.train_on_batch(x=X_batch, y=y_batch)
        
        if epoch % save_interval == 0:
            print ("Epoch: %d | Loss p1: %f | Loss p2: %f" % (epoch, loss1, loss2))


####### -------- Initialize hiperparameters -------- ####### 

LR = 0.0001 # learning rate
opt = tf.keras.optimizers.Adam(learning_rate=LR) # optimizer

####### -------- Initialize players and train model -------- ####### 

# loss_func = tf.keras.losses.Huber()
loss_func = custom_mse_loss

print('Loss function:', loss_func)
print('Initializing players...')
player1, player2 = initialize_players(N, grid_division, loss_func)
print('Begin training!\n')
train(n_epochs, batch_size=BS, prec=grid_division)


'''
# Manual testing
print('\n')
v2 = random.random()
print('v2:', v2)
b2 = np.argmax(player2.predict([v2])) / grid_division
print('\nb2:', b2)
x, y_true = create_batch(b2, batch_size=BS, prec=grid_division)
y_hat = player1.predict(x)
print('\ny_true:', y_true)
print('\ny_hat:', y_hat)
# print( custom_mse_loss(y_true=y_true, y_pred=y_hat) )
print('\n')
'''


####### -------- Play sound when training is finished -------- #######
mixer.init()
mixer.music.load('treinamento_completo.mp3')
mixer.music.play() 


####### -------- Plot results -------- ####### 

x = np.linspace(0, 1, 100)
y1 = [np.argmax(player1.predict([i])) / grid_division for i in x]
y2 = [np.argmax(player2.predict([i])) / grid_division for i in x]
plt.plot(x,y1)
plt.plot(x,y2)
plt.savefig('results/' + str(grid_division) + '_' + str(n_epochs) + '.png')
