#!/usr/bin/env python
# -*- encoding: utf-8 -*-

#~~~~~~~~~~~~~~~~~~~
import sys
sys.path.insert(0, "/home/larz/catkin_chanaka/src/rnn_test01")
#~~~~~~~~~~~~~~~~


import os
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib as mpl

from nnutils.rnn import RNNOfflineData, testRNN # from the nnutils folder rnn.py file

# inside that python file we are going to import RNNOfflineData class and def of testRNN functions.

mpl.use('TkAgg')  # or whatever other backend that you want
if mpl:
    import matplotlib.pyplot as plt

# ⬢⬢⬢⬢⬢➤ DATA
 
number_input_signals = 8
number_output_signals = 15
seq_len = 10 # this will create 1-10 array
# seq_len (10,20) # Start from 10 and run for 20 by one.

# extract
xtestdata_path = "/home/larz/catkin_chanaka/src/rnn_test01/data/xtest.mat" # Need to fix the root of data
xtestdata = loadmat(xtestdata_path)
xTest = xtestdata['test']

ytestdata_path = "/home/larz/catkin_chanaka/src/rnn_test01/data/ytest.mat"
ytestdata = loadmat(ytestdata_path)
yTest = ytestdata['test']

# prepare
rnndata = RNNOfflineData(seq_len, normalise=False, number_input_signals=number_input_signals)
xTest, yTest = rnndata.prepareData(xTest, yTest)


# ⬢⬢⬢⬢⬢➤ TEST


# model = "/home/riccardo/MEGA/test_ml/emg/pippo.h5"
model = "/home/larz/catkin_chanaka/src/rnn_test01/test_ml/emg1/NewModel01.h5"

while True:
    chs = int(input("chs="))
    if chs == -1:
        break
    plotsetting = dict(signals=[chs])
    yPred_test = testRNN(model, xTest, yTest, "test_mat_data", plotsetting=plotsetting)
    plt.show()
