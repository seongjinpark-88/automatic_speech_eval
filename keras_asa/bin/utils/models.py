import os
import pickle

from os.path import isdir, join
from pathlib import Path

import librosa
import librosa.display

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import keras
from keras import optimizers, Input
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, LSTM, GRU, Bidirectional
from keras.utils import to_categorical

from sklearn.model_selection import KFold, train_test_split

def get_audio_features(audio_path):
    """
    Extract mfcc and mel-spectrogram features from audios in the directory
    param: the directory contains the audio file

    return:
            wav2idx: wav2idx[audio_name] = index
            wav_names = wav_names[index] = wav name
            melspec_dict = mel_spectrogram features
            mfcc_dict = mfcc features
    """

    # create empty dictionaries
    wav2idx = {}
    melspec_dict = {}
    mfcc_dict = {}

    # get wav names
    wav_names = [wav for wav in os.listdir(audio_path) if wav.endswith("wav")]

    # set sample length
    samples = 132300
    max_len = 0

    # enumerate over audio files
    for i, w in enumerate(wav_names):

        # get wav idx
        wav2idx[w] = i

        # get the full path of an audio file
        wav_path = audio_path + w
        
        # get sampling rate and audio length
        y, sr = librosa.load(wav_path) 
        
        if 0 < len(y): # workaround: 0 length causes error
            y, _ = librosa.effects.trim(y)

        if len(y) > samples: # long enough
            y = y[0:0+samples]

        else: # pad blank
            padding = samples - len(y)
            offset = padding // 2
            y = np.pad(y, (offset, samples - len(y) - offset), 'constant')
        
        
        mel_data = librosa.feature.melspectrogram(y = y, sr= sr)
        mfcc_data = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 13)
        mfcc_delta = librosa.feature.delta(mfcc_data)
        mfcc_delta2 = librosa.feature.delta(mfcc_data, order = 2)

        mfcc = np.vstack((mfcc_data, mfcc_delta, mfcc_delta2))
        
        melspec_dict[i] = mel_data
        mfcc_dict[i] = mfcc

    return wav2idx, wav_names, melspec_dict, mfcc_dict


def get_data(data_file, wav2idx, feature_dict):
    with open(data_file, "r") as f:
        data = f.readlines()

    X = []
    Y = []
    X_wav = []

    zeros = np.zeros(128)
    for i in range(1, len(data)):
        line = data[i].rstrip()

        accented, stim, wav_name = line.split(",")
        wav_idx = wav2idx[wav_name]
        X_wav.append(wav_idx)

        x_data = feature_dict[wav_idx]

        X.append(x_data)
        Y.append(int(accented)-1)

    X = np.array(X)
    Y = np.array(Y)
    X_wav = np.array(X_wav)

    # print(Y[:10])

    return X, Y, X_wav

def get_cv_index(cv_number, x_input):
    kf = KFold(n_splits = cv_number)
    kf.get_n_splits(x_input)

    return kf.split(x_input)

class Models:
    def __init__(self, X):
        self.input_shape = (np.shape(X)[1], np.shape(X)[2])

    def mlp_model(self, n_connected=2, n_connected_units=32, l_rate=0.001, 
                      dropout=0.2, beta_1=0.9, beta_2=0.999, act='relu',
                      output_act='linear', loss_fx='mean_squared_error',
                      output_size=1):
        """
        Initialize the MLP model
        n_connected:            the number of mlp layers
        n_connected_units:      number of cells in mlp layers
        dropout:                dropout rate in mlp layers
        beta_1:                 value of beta 1 for Adam
        beta_2:                 value of beta 2 for Adam
        act:                    the activation function in lstm + dense layers
        output_act:             the activation function in the final layer
        output_size:            the length of predictions vector; default is 1
        """
        model = Sequential()
        if n_connected >= 3:
            model.add(Dense(n_connected_units, input_dim = self.input_shape, activation = act))
            model.add(Dropout(dropout))
            for i in range(n_connected - 2):
                model.add(Dense(n_connected_units, input_dim=data_shape[-1],
                                     activation=act))
                model.add(Dropout(dropout))
            # add the final layer with output activation
            model.add(Dense(output_size, activation=output_act))
        elif n_nonnected == 2:
            model.add(Dense(n_connected_units, input_dim = self.input_shape, activation = act))
            model.add(Dropout(dropout))
            model.add(Dense(output_size, activation=output_act))

        else:
            # just feed it through linearly if for some reason n_connected = 1
            # this is no longer an mlp, though
            model.add(Dense(output_size, input_dim=self.input_shape,
                            activation=output_act, dropout=dropout))
        # set an optimizer -- adam with default param values
        opt = optimizers.Adam(learning_rate=l_rate)
        # compile the model
        # model.compile(loss=loss_fx, optimizer=opt, metrics=['acc'])
        model.compile(loss=loss_fx, optimizer=opt, metrics=["mse"])
        model.summary()
        return model

    def bi_lstm_model(self, n_lstm=2, n_lstm_units=512, dropout=0.2, n_connected=1,
                         n_connected_units=32, l_rate = 0.001, beta_1=0.9, beta_2=0.999,
                         act='relu', output_act='linear', loss_fx='mse',
                         output_size=1):
        """
        Initialize the LSTM-based model
        n_lstm:                 number of lstm layers
        n_lstm_units:           number of lstm cells in each layer
        dropout:                dropout rate in lstm layers
        n_connected:            the number of fully connected layers
        n_connected_units:      number of cells in connected layers
        beta_1:                 value of beta 1 for Adam
        beta_2:                 value of beta 2 for Adam
        act:                    the activation function in lstm + dense layers
        output_act:             the activation function in the final layer
        output_size:            the length of predictions vector; default is 7
        """
        # clear previously-created model
        # keras.backend.clear_session()

        model = Sequential()
        # add all the hidden layers

        # print("MODEL INFO: ")
        # print("\tN LSTM Layer: " + str(n_lstm))
        # print("\tN LSTM units: " + str(n_lstm_units))
        # print("\tN DENSE Layer: " + str(n_connected))
        # print("\tN DENSE units: " + str(n_connected_units))
        
        if n_lstm > 1:
            model.add(Bidirectional(LSTM(n_lstm_units, return_sequences=True),
                                                  input_shape=self.input_shape))
            for i in range(n_lstm - 1):
                
                # print("Current LSTM layers: " + str(i + 1))
                model.add(Bidirectional(LSTM(n_lstm_units,                                              
                                              return_sequences=False)))
            # print("THE LSTM layers completed")
        else:
            model.add(Bidirectional(LSTM(n_lstm_units, 
                                              return_sequences=False),
                                              input_shape=self.input_shape))
        # add the connected layers
        for i in range(n_connected):
            model.add(Dense(n_connected_units, activation=act))
           
        # print("The connected layer worked")
        # add the final layer with output activation
        model.add(Dense(output_size, activation=output_act))
        # set an optimizer -- adam with default param values
        # print("The output layer worked")
        opt = optimizers.Adam(learning_rate=l_rate)
        # compile the model
        model.compile(loss=loss_fx, optimizer=opt, metrics=["mse"])
        # print("Model compiled")
        model.summary()
        # print("Model compiled successfully")
        return model


    # def cv_train_and_predict(self, batch_size, num_epochs, X, Y, X_wav, wav_names, cv_index, output_name):
    #     CV_mse = []
    #     CV_histories = []
    #     CV_prediction = []

    #     cv_idx = 1
    #     for train_index, test_index in cv_index:

    #         X_train, X_test = X[train_index], X[test_index]
    #         X_train_wav, X_test_wav = X_wav[train_index], X_wav[test_index]
    #         y_train, y_test = Y[train_index], Y[test_index]

    #         model = self.bi_lstm_model(n_connected = 1, n_connected_units = 128)
            
    #         history = model.fit(X_train, y_train, batch_size = batch_size, epochs = num_epochs, shuffle = False,
    #                            class_weight = None, verbose = 1, validation_data = (X_test, y_test))
                               
    #         CV_histories.append(history)
            
    #         scores = model.evaluate(X_test, y_test, verbose=0)
    #         CV_mse.append(scores[1])
    #         y_prediction = model.predict(X_test)
            
    #         for i in range(len(y_test)):
    #             result = "%d\t%s\t%d\t%f\n" % (cv_idx, wav_names[X_test_wav[i]], y_test[i], y_prediction[i][0])
    #     #         print(result)
    #             CV_prediction.append(result)
            
    #         cv_idx += 1

    #         # Final evaluation of the model
    #         scores = model.evaluate(X_test, y_test, verbose=0)
    #         print("MSE: %.5f" % (scores[1]))

    #     with open(output_name, "w") as output:
    #         for prediction in CV_prediction:
    #             output.write(prediction)

    #     return CV_mse, CV_histories, CV_prediction

