import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy.random import choice
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


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
            else: # player loses the auction
                R = 0
            
            # Result of trainable player
            payoffs.append(R)

        y.append(payoffs)

    return np.array(X), np.array(y)


def initialize_players(N, prec=grid_division):
    ''' 
    Initialize players with the same model architecture
    '''
    #@title
    players = []
    for i in range(N):
        model = keras.Sequential([ 
            keras.layers.Flatten(input_shape=(1,)),
            keras.layers.Dense(20, activation=tf.nn.tanh),
            keras.layers.Dense(400, activation=tf.nn.tanh), 
            keras.layers.Dense(100, activation=tf.nn.tanh),
            keras.layers.Dense(prec),
        ])

        model.compile(loss='mse', optimizer=opt)
        players.append(model)
    return players


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


####### -------- Initialize parameters and hiperparameters -------- ####### 

N = 2 # number of players
grid_division = 10 # division of bids in the interval (0,1)
BS = 8 # size of training batch per epoch
LR = 0.001 # learning rate
opt = tf.keras.optimizers.Adam(learning_rate=LR) # optimizer


####### -------- Initialize players and train model -------- ####### 

player1, player2 = initialize_players()
train(5001, batch_size=4, prec=prec)


####### -------- Plot results -------- ####### 

x = np.linspace(0, 1, 100)
y = [np.argmax(player2.predict([i])) / prec for i in x]
plt.plot(x,y)
plt.savefig('first_price_auction.png')