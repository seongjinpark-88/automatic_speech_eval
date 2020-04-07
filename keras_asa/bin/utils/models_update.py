import os, sys
import pickle

from os.path import isdir, join
from pathlib import Path

import librosa
import librosa.display

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import random

seed = 888
random.seed(seed)
np.random.seed(seed)

import tensorflow as tf

import keras
from keras import backend as K
from keras import optimizers

from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten, concatenate
from keras.layers import Conv2D, MaxPooling2D, LSTM, GRU, Bidirectional
from keras.utils import to_categorical

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

def normalize_data(data):
    """
    Normalize the input data to try to avoid NaN output + loss
    From: https://machinelearningmastery.com/how-to-improve-neural-network-\
    stability-and-modeling-performance-with-data-scaling/
    """
    scaler = MinMaxScaler()
    # fit and transform in one step
    normalized = scaler.fit_transform(data)
    # inverse transform
    inverse = scaler.inverse_transform(normalized)
    # return normalized
    return inverse

def pad_feats(feats_list, normalize=False):
    """
    takes the created features dict, ys dit and combines them
    only takes data that has existing x values
    also adds zero padding to the features
    """

    # feats_list = [feats_dict[item] for item in sorted(feats_dict)]

    # if normalize:
    #     feats_list = [normalize_data(item) for item in feats_list]

    if normalize:
        feats_list = [normalize_data(item) for item in feats_list]
    
    padded_feats_list = tf.keras.preprocessing.sequence.pad_sequences(feats_list, padding='post',
                                                                      dtype='float32')

    return np.array(padded_feats_list)

def r_squared(y_true, y_pred):
    """
    r-squared calculation
    from: https://jmlb.github.io/ml/2017/03/20/CoeffDetermination_CustomMetric4Keras/
    """
    ss_res = K.sum(K.square(y_true-y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - (ss_res / (ss_tot + K.epsilon()))

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


def get_data(data_file, wav2idx, feature_dict, acoustic = False):
    with open(data_file, "r") as f:
        data = f.readlines()

    X = []
    Y = []
    X_wav = []

    zeros = np.zeros(128)
    for i in range(1, len(data)):
        line = data[i].rstrip()

        # accented, stim, wav_name = line.split(",")
        stim, spk, accented = line.split(",")
        wav_name = stim + ".wav"
        wav_idx = wav2idx[wav_name]
        X_wav.append(wav_idx)
        if acoustic == False:
            x_data = feature_dict[wav_idx]
            X.append(x_data)
        else: 
            x_data = feature_dict[stim]
            # x_data = np.array(x_data)
            # x_data = np.reshape(x_data, np.shape(x_data)[1])
            X.append(x_data)

        Y.append(float(accented)-1)

    X = np.array(X)
    Y = np.array(Y)
    X_wav = np.array(X_wav)

    print(Y[:10])

    return X, Y, X_wav

def get_cv_index(cv_number, x_input):
    kf = KFold(n_splits = cv_number)
    kf.get_n_splits(x_input)

    return kf.split(x_input)

class Models:
    def __init__(self, X, name, model_type):
        if model_type == "lstm":
            self.input_shape = (np.shape(X)[1], np.shape(X)[2])
        elif model_type == "mlp":
            self.input_shape = (np.shape(X)[-1],)
        else:
            print("Undefined input shape")
            exit()

        self.name = name

    def mlp_model(self, n_connected_units=32, dropout=0.2, act='tanh'):
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

        self.mlp_input = Input(shape = self.input_shape, name=self.name)

        second_units = int(n_connected_units / 2)
        
        output_1 = Dense(n_connected_units, activation = act)(self.mlp_input)
        dropout_1 = Dropout(dropout)(output_1)
        output_2 = Dense(second_units, activation = act)(dropout_1)
        dropout_2 = Dropout(dropout)(output_2)
        final_dense = Dense(1, activation = 'linear')(dropout_2)
        return final_dense

    def bi_lstm_model(self, n_lstm_units=512, dropout=0.2, n_connected_units=32, act='tanh'):
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

        self.lstm_input = Input(shape = self.input_shape, name=self.name)
  
        output_1 = Bidirectional(LSTM(n_lstm_units, activation = act, 
            dropout = dropout, recurrent_dropout = dropout, return_sequences = True))(self.lstm_input)
        output_2 = Bidirectional(LSTM(n_lstm_units, activation = act, 
            dropout = dropout, recurrent_dropout = dropout, return_sequences = False))(output_1)
        
        output_3 = Dense(n_connected_units, activation = 'tanh')(output_2)
        dropout_3 = Dropout(dropout)(output_3)
        final_dense = Dense(1, activation = 'linear')(dropout_3)
        return final_dense


class MergeModels:
    """
    Takes different models, merge it, and train/test the models
    Each model should have same order of data
    """

    def __init__(self, input_models, input_layers):
        self.merged_layers = concatenate(input_models)
        self.input_layers = input_layers

    def final_layers(self, n_connected = 1, n_connected_units = 36, 
        dropout = 0.2, act = 'tanh', output_act = 'linear', loss_fx = 'mse'):
        second_units = int(n_connected_units / 2)
        
        # dense_1 = Dense(n_connected_units, activation = act)(self.merged_layers)
        # dropout_1 = Dropout(dropout)(dense_1)

        # dense_2 = Dense(second_units, activation = act)(dropout_1)
        # dropout_2 = Dropout(dropout)(dense_2)
        
        self.final_output = Dense(1, activation = output_act, name = 'final_output')(self.merged_layers)

    def compile_model(self, l_rate = 0.001, loss_fx='mse', beta_1 = 0.9, beta_2 = 0.999):
        self.model = Model(inputs = self.input_layers, outputs = [self.final_output])
        # opt = optimizers.Adam(learning_late = l_rate, beta_1 = beta_1, beta_2 = beta_2)
        self.model.compile(optimizer="adam", loss = loss_fx, metrics = ['mse', r_squared])
        self.model.summary()

    def train_model(self, epochs = 100, batch_size = 64, input_feature = None, 
        output_label = None, validation = None, model_name = None):

        early_stopping = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10)

        model_name = model_name + ".h5"

        save_best = ModelCheckpoint(model_name, monitor = 'val_loss', mode = 'min')

        self.history = self.model.fit(input_feature, {'final_output': output_label}, 
            epochs = epochs, batch_size = batch_size, shuffle = True, 
            # validation_data = validation, verbose = 1, 
            validation_split = 0.1, verbose = 1,
            callbacks = [early_stopping, save_best])
        return self.history

    def save_model(self, save_path):
        self.model.save(save_path)

    def load_model(self, model_path):
        return load_model(model_path)

    def predict_model(self, input_feature):
        self.pred = self.model.predict(input_feature)
        return self.pred

    def evaluate_model(self, input_feature, true_label):
        self.scores = self.model.evaluate(input_feature, {'final_output': true_label}, verbose = 0)
        return self.scores


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
                        os.system("{0}/SMILExtract -loglevel 0 -C {0}/config/IS09_emotion.conf -I {1}/{2}\
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