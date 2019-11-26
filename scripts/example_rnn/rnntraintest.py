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

from nnutils.rnn import RNNOfflineData, buildRNN, testRNN
# from nnutils.rnn import buildRNN
# from nnutils.rnn import testRNN

mpl.use('TkAgg')  # or whatever other backend that you want
if mpl:
    import matplotlib.pyplot as plt


##########################################################################
##########################################################################
##########################################################################


# ⬢⬢⬢⬢⬢➤ DATA

plot_input_data_flg = True #False
number_input_signals = 8
number_output_signals = 15
seq_len = 10  # 5

# ➤➤➤➤➤ Extract DATA

# input
# xdata_path = "/home/riccardo/tests/test_emg/data/emg.mat"
xtraindata_path = "/home/larz/catkin_chanaka/src/rnn_test01/data/xtrain.mat"
xtraindata = loadmat(xtraindata_path)
xTrain = xtraindata['train']  # emg

xtestdata_path = "/home/larz/catkin_chanaka/src/rnn_test01/data/xtest.mat"
xtestdata = loadmat(xtestdata_path)
xTest = xtestdata['test']  # emg

# output
# ydata_path = "/home/riccardo/tests/test_emg/data/leap.mat"
ytraindata_path = "/home/larz/catkin_chanaka/src/rnn_test01/data/ytrain.mat"
ytraindata = loadmat(ytraindata_path)
yTrain = ytraindata['train']  # joints

ytestdata_path = "/home/larz/catkin_chanaka/src/rnn_test01/data/ytest.mat"
ytestdata = loadmat(ytestdata_path)
yTest = ytestdata['test']  # emg

data_out_training_dict = {}
data_out_training_dict["yTest"] = yTest

# ➤➤➤➤➤ Prepare DATA

rnndata = RNNOfflineData(seq_len, normalise=False, number_input_signals=number_input_signals)
xTrain, yTrain = rnndata.prepareData(xTrain, yTrain)
xTest, yTest = rnndata.prepareData(xTest, yTest)
data_out_training_dict["xTrain"] = xTrain
data_out_training_dict["yTrain"] = yTrain
data_out_training_dict["yTest"] = yTest
data_out_training_dict["xTest"] = xTest
print("xTrain = {}\nyTrain ={}\nxTest ={}\nyTest ={}\n\n\n".format(xTrain.shape, yTrain.shape,  xTest.shape, yTest.shape))

# ➤➤➤➤➤ Plot DATA
if plot_input_data_flg:
    plt.figure()
    for i in range(seq_len):
        plt.plot(np.reshape(xTrain[:, i, :], (xTrain.shape[0], 8)), label="xTrain")
    plt.figure()
    # plt.plot(yTrain, label="yTrain")
    plt.legend()
    plt.show()

##########################################################################
##########################################################################
##########################################################################

# ⬢⬢⬢⬢⬢➤ TRAINING

# PARAMETRS
save_model_flg = True # False
save_model_path = "/home/larz/catkin_chanaka/src/rnn_test01/test_ml/emg1"
model_name = "NewModel_traintest01"

lstm_cells = [100, 100, 100, 100, 100]
dropout_value = [0.5, 0.5, 0.5, 0.5, 0.5]
dense_neurons = [16, 16, 16,  number_output_signals]

epochs = 1
learning_rate = 1e-3
rho = 0.9
decay = 0
validation_split = 0.1


# ⬢⬢⬢⬢⬢➤ MODEL

# BUILD
rnnmodel = buildRNN(
    number_input_signals=number_input_signals,
    number_output_signals=number_output_signals,
    seq_len=seq_len,
    dropout_value=dropout_value,
    lstm_cells=lstm_cells,
    dense_activation=None,
    dense_neurons=dense_neurons
)

# OPTIMIZATION SETTINGS
rmsprop = keras.optimizers.RMSprop(
    lr=learning_rate,
    rho=rho,  # decay factor average over the square of the gradients
    epsilon=None,
    decay=decay  # decays the learning rate over time, so we can move even closer to the local minimum in the end of training
)
rnnmodel.compile(
    loss="mse",
    optimizer=rmsprop
)
print("I'm in 129")
# ⬢⬢⬢⬢⬢➤ FIT
rnnmodel.fit(
    xTrain,
    yTrain,
    nb_epoch=epochs,
    verbose=1,
    shuffle=True,
    validation_split=validation_split
)
print("I'm in 140")

if save_model_flg:
    rnnmodel.save(os.path.join(save_model_path, '{}.h5'.format(model_name)))

##########################################################################
##########################################################################
##########################################################################
print("I'm in 147 save done")
# ⬢⬢⬢⬢⬢➤ TEST on testData
while True:
    chs = int(input("chs="))
    if chs == -1:
        break
    plotsetting = dict(signals=[chs])
    yPred_test = testRNN(rnnmodel, xTest, yTest, "test_mat_data", plotsetting=plotsetting)
    plt.show()

xTest = xTrain  # emg
yTest = yTrain  # joints

print("I'm in 160 done in testdata")
# ⬢⬢⬢⬢⬢➤ TEST on trainData
while True:
    chs = int(input("chs="))
    if chs == -1:
        break
    plotsetting = dict(signals=[chs])
    yPred_test = testRNN(rnnmodel, xTest, yTest, "test_mat_data", plotsetting=plotsetting)
    plt.show()

print("I'm in 160 done in traindata")

data_out_training_dict["yPred_test"] = yPred_test

savemat(os.path.join(save_model_path, 'outputData01.mat'), data_out_training_dict)
# outputData01.mat is the file we are going to save

print("All done")