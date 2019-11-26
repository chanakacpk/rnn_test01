#!/usr/bin/env python
# -*- encoding: utf-8 -*-

#~~~~~~~~~~~~~~~~~~~
import sys
sys.path.insert(0, "/home/larz/catkin_chanaka/src/rnn_test01")
#~~~~~~~~~~~~~~~~

import keras
import os
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib as mpl

from nnutils.rnn import RNNOfflineData, buildRNN

mpl.use('TkAgg')  # or whatever other backend that you want
if mpl:
    import matplotlib.pyplot as plt


##########################################################################
##########################################################################
##########################################################################

# ⬢⬢⬢⬢⬢➤ DATA~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

number_input_signals = 8
number_output_signals = 15
seq_len = 10  # number_of_time_steps
plot_input_data_flg = False

print("I'm in line 33") #For testing purpose
# ➤➤➤➤➤ Extract DATA
# input
# xdata_path = "/home/riccardo/tests/test_emg/data/emg.mat"
xtraindata_path = "/home/larz/catkin_chanaka/src/rnn_test01/data/xtrain.mat"
xtraindata = loadmat(xtraindata_path)
xTrain = xtraindata['train']  # emg

print("I'm in line 42 XTrain data loaded") #For testing purpose
# output
# ydata_path = "/home/riccardo/tests/test_emg/data/leap.mat"
ytraindata_path = "/home/larz/catkin_chanaka/src/rnn_test01/data/ytrain.mat"
ytraindata = loadmat(ytraindata_path)
yTrain = ytraindata['train']  # joints

print("I'm in line 48 YTrain data loaded") #For testing purpose

# ➤➤➤➤➤ Prepare DATA
rnndata = RNNOfflineData(seq_len, normalise=False, number_input_signals=number_input_signals)
xTrain, yTrain = rnndata.prepareData(xTrain, yTrain)
print("xTrain = {}\nyTrain ={} \n\n\n".format(xTrain.shape, yTrain.shape))

# ➤➤➤➤➤ Plot DATA
if plot_input_data_flg:
    plt.figure()
    for i in range(seq_len):
        plt.plot(np.reshape(xTrain[:, i, :], (xTrain.shape[0], 8)), label="xTrain")
    plt.figure()
    # plt.plot(yTrain, label="yTrain")
    plt.legend()
    plt.show()

print("I'm in line 65 plot data done") #For testing purpose
##########################################################################
##########################################################################
##########################################################################

# ⬢⬢⬢⬢⬢➤ TRAINING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# PARAMETRS
save_model_flg = True #False
save_model_path = "/home/larz/catkin_chanaka/src/rnn_test01/test_ml/emg1"
model_name = "NewModel01"

lstm_cells = [100, 100, 100, 100, 100]
dropout_value = [0.5, 0.5, 0.5, 0.5, 0.5]
dense_neurons = [16, 16, 16,  number_output_signals]

epochs = 12
learning_rate = 1e-3
rho = 0.9  # decay factor average over the square of the gradients
decay = 0  # decays the learning rate over time, so we can move even closer to the local minimum in the end of training
validation_split = 0.05

print("I'm in line 87 parameters done") #For testing purpose
# ⬢⬢⬢⬢⬢➤ MODEL

# BUILD~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
rnnmodel = buildRNN(
    number_input_signals=number_input_signals,
    number_output_signals=number_output_signals,
    seq_len=seq_len,
    dropout_value=dropout_value,
    lstm_cells=lstm_cells,
    dense_activation=None,
    dense_neurons=dense_neurons
)

print("I'm in line 101. build model is done") #For testing purpose
# OPTIMIZATION SETTINGS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
rmsprop = keras.optimizers.RMSprop(
    lr=learning_rate,
    rho=rho,
    epsilon=None,
    decay=decay
)

rnnmodel.compile(
    loss="mse",
    optimizer=rmsprop
)
# """
# optimizer: An optimizer. It can be the string identifier of an existing optimizer
# (e.g. rmsprop or adagrad), or an instance of the Optimizer class.  
# loss: A loss function (the objective that the model will try to minimize). 
# It can be the string identifier of an existing loss function 
# (e.g. categorical_crossentropy or mse), or it can be an objective function.  
# metrics: A list of metrics.  It can be the string identifier of an existing
# metric or a custom metric function. For any classification problem you will
# want to set this to metrics=['accuracy']. 
# """

print("I'm in line 125 optimization done") #For testing purpose
# ⬢⬢⬢⬢⬢➤ FIT~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
rnnmodel.fit(
    xTrain,
    yTrain,
    nb_epoch=epochs,
    verbose=1,
    shuffle=True,
    validation_split=validation_split
)
print("I'm fit model done  in line 135") #For testing purpose

####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Testing 
# Evaluate performance in one line 

loss_and_matrics = rnnmodel.Evaluate(xTrain, yTrain)

# Generate predictions on new data

classes = rnnmodel.predict(x_Test, batch_size=128)

####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


if save_model_flg:
    rnnmodel.save(os.path.join(save_model_path, '{}.h5'.format(model_name)))
print("I'm fit model done  in last save_model_flg done.") #For testing purpose


# Huraaaaay