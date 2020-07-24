# Setup
import numpy as np
from convnet import *
from all_layers import *
from solver_trainer import Solver

"""
This file implements the main task of the challenge. We import training and test data
from .csv files. Then process the data for faster convergece.Then reshape them
into original images with depth=1. We also split the the training data into training and
validation sets. Then we create a object of the three layer neural network class, with
chosen number of filters, their size and number of hidden units (initialiser
also randomly initialises the parameters). Then we train the model for chosen number of
epochs, batch size, reglarisation strength and learning rate. All the hyperparameters used
here have been chosen so as to increase "validation accuracy", which has been done over
many cycles of model training.

"""

#%%
# Importing training(will be split into training and validation) and test data
data_mnist = {}
train_data = np.genfromtxt("train.csv", delimiter=',')
data_mnist['X_train'] = np.zeros((train_data.shape[0]-1-1000,1,28,28))
data_mnist['y_train'] = train_data[1001:train_data.shape[0],0]

data_mnist['X_val'] = np.zeros((1000,1,28,28))
data_mnist['y_val'] = train_data[1:1001,0]

test_data = np.genfromtxt("test.csv", delimiter=',')
data_mnist['X_test'] = np.zeros((test_data.shape[0]-1,1,28,28))
#%%
# Preprocessing: Subtracting mean across all training images for each pixel
# to run ONLY ONCE
mean = np.mean(train_data[1001:train_data.shape[0], 1:train_data.shape[1]], axis=0)

train_data[1:train_data.shape[0], 1:train_data.shape[1]] = train_data[1:train_data.shape[0], 1:train_data.shape[1]] - mean

test_data[1:test_data.shape[0],:] = test_data[1:test_data.shape[0],:] - mean
#%%
# Reshaping each row into image of depth=1, height=28, width=28

for i in range(train_data.shape[0]-1):
    if i<1000:
        data_mnist['X_val'][i,0,:,:] = np.reshape(train_data[i+1,1:train_data.shape[1]],(28,28))
    else:
        data_mnist['X_train'][i-1000,0,:,:] = np.reshape(train_data[i+1,1:train_data.shape[1]],(28,28))

for i in range(test_data.shape[0]-1):
    data_mnist['X_test'][i,0,:,:] = np.reshape(test_data[i+1,:],(28,28))


data_mnist['y_train'] = data_mnist['y_train'].astype('int8')
data_mnist['y_val'] = data_mnist['y_val'].astype('int8')


for k, v in data_mnist.items():
  print('%s: ' % k, v.shape)

#%%
# training the model with choice of hyperparameters
model = ThreeLayerConvNet(num_filters=5, filter_size=3,
                          input_dim=(1,28,28), hidden_dim=25, num_classes=10, reg=1e-8,
                          dtype=np.float64)

solver = Solver(model, data_mnist, batch_size=50, num_epochs=5, optim_type="adam", optim_config={'learning_rate': 1e-3,}, lr_decay=np.sqrt(0.1), num_train_samples=1000, num_val_samples=None, verbose=True)
solver.training()


#%%

# Print validation accuracy over entire validation set
print(
    "Full data training accuracy:",
    solver.accuracy(data_mnist['X_val'], data_mnist['y_val'])
)
#%%

# get test data predictions

y_pred = solver.accuracy(data_mnist['X_test'], None)
print(y_pred)

#%%
# convert predictions to .csv file

import pandas as pd
dt = pd.DataFrame(data=y_pred)
dt.to_csv('predictions.csv', mode='a', index=True)
#%%
