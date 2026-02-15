import os, sys
import pickle

from os.path import isdir, join
from pathlib import Path

import librosa
import librosa.display

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras import optimizers, Input
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, LSTM, GRU, Bidirectional
from keras.utils import to_categorical

from sklearn.model_selection import KFold, train_test_split

def r_squared(y_true, y_pred):
    """
    r-squared calculation
    from: https://jmlb.github.io/ml/2017/03/20/CoeffDetermination_CustomMetric4Keras/
    """
    ss_res = K.sum(K.square(y_true-y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())

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
        if len(np.shape(X) > 2):
            self.input_shape = (np.shape(X)[1], np.shape(X)[2])
        elif len(np.shape(X) == 2):
            self.input_shape = (np.shape(X)[1])

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
            model.add(Bidirectional(LSTM(n_lstm_units, activation=act, 
                                            dropout=dropout, recurrent_dropout=dropout,
                                            return_sequences=True),
                                                  input_shape=self.input_shape))
            for i in range(n_lstm - 1):
                
                # print("Current LSTM layers: " + str(i + 1))
                model.add(Bidirectional(LSTM(n_lstm_units, activation=act, 
                                            dropout=dropout, recurrent_dropout=dropout,                                     
                                              return_sequences=False)))
            # print("THE LSTM layers completed")
        else:
            model.add(Bidirectional(LSTM(n_lstm_units, activation=act, 
                                            dropout=dropout, recurrent_dropout=dropout,
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
        opt = optimizers.Adam(learning_rate=l_rate, beta_1 = beta_1, beta_2 = beta_2)
        # compile the model
        model.compile(loss=loss_fx, optimizer=opt, metrics=[r_squared])
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

class GetFeatures:
    """
    Takes input files and gets segmental and/or suprasegmental features
    Current features extracted: XXXX, YYYY, ZZZZ
    """
    def __init__(self, audio_path, opensmile_path, save_path):
        self.apath = audio_path
        self.smilepath = opensmile_path
        self.savepath = save_path
        self.supra_name = None # todo: delete?
        self.segment_name = None # todo: delete?

    #
    # def copy_files_to_single_directory(self, single_dir_path):
    #     """
    #     Copy all files for different speakers to a single directory
    #     single_dir_path : full path
    #     """
    #     if not os.path.isdir(single_dir_path):
    #         os.system("mkdir {0}".format(single_dir_path))
    #     # for f in os.scandir(self.apath):
    #     #     if f.is_dir() and str(f).startswith("S"):
    #     #         print(f)
    #     os.system("cp -r {0}/S*/wav/* {2}/".format(self.apath, single_dir_path))
    #     self.apath = single_dir_path

    def extract_features(self, supra=False, summary_stats=False):
        """
        Extract the required features in openSMILE
        """
        # for file in directory
        for f in os.listdir(self.apath):
            # get all wav files
            if f.endswith('.wav'):
                wavname = f.split('.')[0]
                # extract features
                # todo: replace config files with the appropriate choice
                if supra is True:
                    os.system("{0}/SMILExtract -C {0}/config/IS10_paraling.conf -I {1}/{2}\
                          -lldcsvoutput {3}/{4}.csv".format(self.smilepath, self.apath, f,
                                                            self.savepath, wavname))
                    # self.supra_name = output_name # todo: delete?
                else:
                    if summary_stats is False:
                        os.system("{0}/SMILExtract -loglevel 0 -C {0}/config/IS09_emotion.conf -I {1}/{2}\
                              -lldcsvoutput {3}/{4}.csv".format(self.smilepath, self.apath, f,
                                                                self.savepath, wavname))
                    else:
                        os.system("{0}/SMILExtract -loglevel 0 -C {0}/config/IS10_paraling.conf -I {1}/{2}\
                              -csvoutput {3}/{4}.csv".format(self.smilepath, self.apath, f,
                                                             self.savepath, wavname))
                    # self.segment_name = output_name # todo: delete?

    def get_features_dict(self, dropped_cols=None):
        """
        Get the set of phonological/phonetic features
        """
        # create a holder for features
        feature_set = {}

        # iterate through csv files created by openSMILE
        for csvfile in os.listdir(self.savepath):
            if csvfile.endswith('.csv'):
                csv_name = csvfile.split(".")[0]
                # get data from these files
                csv_data = pd.read_csv("{0}/{1}".format(self.savepath, csvfile), sep=';')
                # drop name and time frame, as these aren't useful
                if dropped_cols:
                    csv_data = self.drop_cols(csv_data, dropped_cols)
                else:
                    csv_data = csv_data.drop(['name', 'frameTime'], axis=1).to_numpy().tolist()
                if "nan" in csv_data or "NaN" in csv_data or "inf" in csv_data:
                    pprint.pprint(csv_data)
                    print("Data contains problematic data points")
                    sys.exit(1)

                # add it to the set of features
                feature_set[csv_name] = csv_data

        return feature_set

    def drop_cols(self, dataframe, to_drop):
        """
        to drop columns from pandas dataframe
        used in get_features_dict
        """
        return dataframe.drop(to_drop, axis=1).to_numpy().tolist()