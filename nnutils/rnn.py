#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import warnings
import math
import sys
import time
import numpy as np
import keras
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.models import load_model, Sequential
from keras.utils import plot_model
import tensorflow as tf
import matplotlib as mpl


################# 15112019
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
##################### Warning error solved by above code for more info visit
# https://github.com/CyberZHG/keras-bert/issues/139
# https://docs.google.com/document/d/1xNnDsrVwE331Xjbe4-xNOTGFPQaJZBELzF_lAOs5x1g/edit?usp=sharing



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide messy TensorFlow warnings
# os.environ will get the user environment
warnings.filterwarnings("ignore")  # Hide messy Numpy warnings
# If your script depends on a specific backend you can use the use() function:

mpl.use('TkAgg')  # or whatever other backend that you want
# If you use the use() function, this must be done before importing matplotlib.pyplot. 
# Calling use() after pyplot has been imported will have no effect. 
# Using use() will require changes in your code if users want to use a different backend.
# Therefore, you should avoid explicitly calling use() unless absolutely necessary.

if mpl:
    import matplotlib.pyplot as plt

##########################################################################
##########################################################################
##########################################################################
''' MODEL '''


def buildRNN(number_input_signals, number_output_signals, seq_len, dropout_value, lstm_cells, dense_neurons, dense_activation="tanh"):
    # activation_functions -> sigmoid function, Rectifier function and Hyperbolic Tangent (tanh) function
    # for hidden layer "Rectifier activation" is best and for output layer "Sigmoid activation" is good.
    lstm_layers = len(lstm_cells)
    dense_layers = len(dense_neurons)
    rs = lstm_layers > 1

    model = Sequential() # The first layer in a Sequential model needs to receive information about its input shape. 

    model.add(LSTM(units=lstm_cells[0],
                   input_shape=(seq_len, number_input_signals),
                   return_sequences=rs)) # That return sequences return the hidden state output for each input time step.
    model.add(Dropout(dropout_value[0]))
# input_shape: this is a shape tuple (the batch dimension is not included).
# seq_len = number_of_time_steps
# input_dim: number of inputs values (or input vector dimension) supported in some 2D layers (e.g. Dense)
# input_length: number of time steps in the sequence supported in some 3D temporal layers (e.g. LSTM)
# batch_size: defines number of samples that going to be propagated through the network (this is useful for stateful recurrent networks). 

    if rs:
        for layer_index in range(lstm_layers - 2):
            model.add(LSTM(units=lstm_cells[layer_index],
                           input_shape=(seq_len, number_input_signals),
                           return_sequences=True))
            model.add(Dropout(dropout_value[layer_index]))

        model.add(LSTM(units=lstm_cells[-1],
                       input_shape=(seq_len, number_input_signals),
                       return_sequences=False))
        model.add(Dropout(dropout_value[-1]))

    for layer_index in range(dense_layers):
        model.add(Dense(dense_neurons[layer_index], activation=dense_activation))

    return model


##########################################################################
##########################################################################
##########################################################################
''' PREDICTION '''


class RNN(object):

    def __init__(self, model_h5_file_path, seq_len, bias_window_size=10,  num_input_signals=16):
        self.data = RNNOnlineData(seq_len)
        self.pred_history = np.zeros(bias_window_size)
        self.bias_window_size = bias_window_size
        self.init_ct = bias_window_size
        self.model = None
        self.reg_name = model_h5_file_path.split("/")[-1]
        try:
            self.model = load_model(model_h5_file_path)
            self.model.predict(np.zeros((1, seq_len, num_input_signals)))  # void prediction to create the graph (multithread problem in tensorflow with GPU)
            # self.model._make_predict_function() # https://github.com/fchollet/keras/issues/2397, https://github.com/fchollet/keras/issues/6124
            # self.graph = tf.get_default_graph()
            print("Regressors found and loaded!\n" + model_h5_file_path)
        except Exception as e:
            print("None regressor found!\n " + model_h5_file_path)
            print(e)

    def update(self, msg):
        self.data.update(msg)
        if not self.isReady():
            self.predict()  # for initialization purpose

    def predict(self):
        pred = self.model.predict(self.data.getWindow())
        pred = pred.item()  # np.ravel(pred, order='F') # flatten the prediction in a single-element-array
        self._update_history(pred)
        return pred

    def _update_history(self, last_pred):
        tmp = np.append(self.pred_history, [last_pred], axis=0)  # add the new prediction in the queue
        self.pred_history = np.delete(tmp, 0, 0)  # remove the oldes prediction from the queue
        if not self.isReady():
            print("{} initialization {}".format(self.reg_name, self.init_ct))
            self.init_ct -= 1

    def reset(self):
        self.init_ct = self.bias_window_size

    def isReady(self):
        return self.init_ct <= 0

    def getMean(self):
        mean = sum(self.pred_history) / len(self.pred_history)
        return mean


##########################################################################
def testRNN(model, xTest, yTest, name, plotsetting=None):
    ''' @model can be either the .h5 model or the path (str) to the model'''

    if type(model) is str:
        model = load_model(model)
    test_data_lenght = yTest.shape[0]
    number_output_signals = yTest.shape[1]
    yPred = model.predict(xTest)
    yPred = np.reshape(yPred, (test_data_lenght, number_output_signals))
    print("\nxTest ={}\nyTest ={}\nyPred = {}".format(xTest.shape, yTest.shape, yPred.shape))

    if plotsetting is not None:
        try:
            a = np.zeros([test_data_lenght, len(plotsetting["signals"])])
            b = np.zeros([test_data_lenght, len(plotsetting["signals"])])
            for i, j in enumerate(plotsetting["signals"]):
                a[:, i] = yPred[:, j]
                b[:, i] = yTest[:, j]
            yPred = a
            yTest = b
        except:
            print("@@@  ERROR  @@@")
            pass
    print("\nPLOT INFO: \t yTest ={}\nyPred = {}".format(yTest.shape, yPred.shape))

    plt.figure()
    plt.plot(yPred, label="yPred")
    plt.plot(yTest, label="yTest")
    plt.legend()
    plt.title(name)

    return yPred


##########################################################################
##########################################################################
##########################################################################
''' data sequence for training '''


class RNNOfflineData(object):
    def __init__(self, seq_len, normalise=False, number_input_signals=None):
        self.seq_len = seq_len
        self.normalise = normalise
        self.number_input_signals = number_input_signals

    def groupDataWindow(self, raw_data): #inputs "self and raw_data"
        data_flow_length = len(raw_data)
        data = [] # Make a dictionary called data
        for index in range(data_flow_length - self.seq_len):
            window_data = raw_data[index: index + self.seq_len, :]
            data.append(window_data) # add new data end of the list
        if self.normalise and self.number_input_signals is not None:
            data = self.normalise_windows(data)
        data = np.array(data)
        return data

    def prepareData(self, x, y,  axis=None):
        if axis is not None:
            y = y[:, axis]
        x = self.groupDataWindow(x)
        y = y[self.seq_len:]
        y = np.reshape(y, (y.shape[0], y.shape[1]))
        return x, y

    def normalise_windows(self, window_data_list):
        # reflect percentage changes from the start of that window
        # (so the data at point i=0 will always be 0)
        normalised_window_data_list = []
        for window_data in window_data_list:
            normalised_window_data = np.zeros((self.seq_len, self.number_input_signals))
            for j in range(self.seq_len):
                for i in range(self.number_input_signals):
                    w0 = window_data[0][i]
                    normalised_window_data[j, i] = (float(window_data[j][i]) / float(w0)) - 1
            normalised_window_data_list.append(normalised_window_data)
        return normalised_window_data_list


##########################################################################
##########################################################################
##########################################################################
''' data stream for prediction '''


class RNNOnlineData(object):

    def __init__(self, seq_len):
        self.window = np.zeros((seq_len, 16))
        self.new_sample = None

    def update(self, msg):
        self.new_sample = np.ravel(msg.data).reshape((1, 16))
        self._window_update(self.new_sample)

    def _window_update(self, new_sample):
        tmp_window = np.append(self.window, new_sample, axis=0)
        self.window = np.delete(tmp_window, 0, 0)

    def getWindow(self):
        w = np.reshape(self.window, (1, self.window.shape[0], self.window.shape[1]))  # vector -> tensor
        return w
