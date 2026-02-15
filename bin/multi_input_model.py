# prepare model that accepts multiple types of input

import os, sys
from collections import OrderedDict
import operator
from functools import reduce

import numpy as np
import pandas as pd
import pickle
import random
import pprint
import warnings
import h5py

# set seed for reproducibility
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.engine import Layer

seed = 888
random.seed(seed)
np.random.seed(seed)

import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Bidirectional, Dropout, Flatten, TimeDistributed, Masking
from keras import optimizers, Input
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# todo: finish data prep code, test on phonetic features!
# todo: ensure that we can actually get the feature sets we want -- work on this
# todo: rename suprasegmental/segmental to phonological/phonetic, respectively


def r_squared(y_true, y_pred):
    """
    r-squared calculation
    from: https://jmlb.github.io/ml/2017/03/20/CoeffDetermination_CustomMetric4Keras/
    """
    ss_res = K.sum(K.square(y_true-y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())


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
    return inverse


def reshape_data(data, n_dim=2):
    """
    To convert 3d data to 2d data (bc of how data was extracted)
    n_dim : number of required dimensions
    """
    if n_dim != 2:
        return "Data should not be reshaped or can't be at this time"
    else:
        return np.reshape(data, (data.shape[0], data.shape[-1]))


def get_phonological_features(setpath):
    """
    Get the phonological features from a csv file
    """
    phon_dict = {}
    with open(setpath, 'r') as phonfile:
        for line in phonfile:
            line = line.strip().split(',')
            wav_name = line[0].split('.')[0]
            data = line[1:]
            phon_dict[wav_name] = data
    return phon_dict

def get_ys_dict_all(ypath, speaker_list):
    """
    get the set of y values for the data;
    these come from a csv with 3 cols:
    score, stimuli, wav
    ypath: the path to the csv, INCLUDING file name
    """
    ys = {}
    with open(ypath, 'r') as yfile:
        for line in yfile:
            line = line.strip().split(",")
            (speaker, sentence) = line[1].split("_")

            if speaker in speaker_list:
                ys[line[1]] = line[0]
    return ys


def get_ys_dict(ypath, speaker_list):
    """
    get the set of y values for the data;
    these come from a csv with 3 cols:
    stimulus, speaker, average_score
    ypath: the path to the csv, INCLUDING file name
    """
    ys = {}
    with open(ypath, 'r') as yfile:
        for line in yfile:
            line = line.strip().split(",")
            if line[1] in speaker_list:
                ys[line[0]] = line[2]
    return ys


def zip_feats_and_ys(feats_dict, ys_dict, normalize=False):
    """
    takes the created features dict, ys dit and combines them
    only takes data that has existing x values
    also adds zero padding to the features
    """
    for item in sorted(feats_dict.keys()):
        if item not in ys_dict.keys():
            feats_dict.pop(item)
    # normalize feats list
    if normalize:
        feats_list = [normalize_data(feats_dict[item]) for item in sorted(feats_dict)]
    else:
        feats_list = [feats_dict[item] for item in sorted(feats_dict)]
    # print(feats_list[0])
    ys_list = [float(ys_dict[item]) for item in sorted(ys_dict)]
    padded_feats_list = tf.keras.preprocessing.sequence.pad_sequences(feats_list, padding='post',
                                                                      dtype='float32')
    # print(ys_list[0])
    return zip(padded_feats_list, ys_list)


def train_and_predict(model, trainX, trainy, valX, valy, batch=32, num_epochs=100,
                      cv=False):  # , savefile='phonological_cv_log.csv'):
    """
    Train the model
    batch:              minibatch size
    num_epochs:         number of epochs
    """
    # create early stopping criterion -- stops when val_loss starts to increase
    if cv:
        early_stopping = EarlyStopping(monitor='loss', mode='min', patience=10)
        # save_best = ModelCheckpoint('best_cv.h5', monitor='loss', mode='min')
    else:
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=10)
    # save best model
    #     save_best = ModelCheckpoint('best.h5', monitor='val_loss', mode='min')
    # csv_saver = CSVLogger(savefile, append=True, separator=',')
    model.fit(trainX, trainy, batch_size=batch, epochs=num_epochs, shuffle=True,
              class_weight=None, validation_data=(valX, valy),
              callbacks=[early_stopping])  # , save_best , csv_saver])
    # # get summary of model
    # model.summary()
    # sys.exit(1)
    # get predictions on the dev set
    y_preds = model.predict(valX, batch_size=batch)

    return valy, y_preds


def save_model(model, m_name='best_model.h5'):
    model.save(m_name)


def cv_train_wrapper(model, cv_data, cv_ys, batch, num_epochs):  # , savefile="phonological_cv_log.csv"):
    """
    Wrapper for training with k-fold CV
    returns all the predictions for y along with all gold y values
    """
    all_y = []
    all_preds = []
    cv_data = OrderedDict(cv_data)
    cv_ys = OrderedDict(cv_ys)
    for key, val in cv_data.items():
        valX = np.asarray(cv_data[key])
        # print(valX)
        # print("END OF VAL X AND START OF TRAIN X")
        valy = np.asarray(cv_ys[key])
        trainX = np.asarray([item for k in set(cv_data.keys()-[key]) for item in cv_data[k]])
        # print(trainX)
        # sys.exit(1)
        trainy = np.asarray([item for k in set(cv_ys.keys()-[key]) for item in cv_ys[k]])
        valy, ypreds = train_and_predict(model, trainX, trainy, valX, valy, batch, num_epochs, cv=True)  # , savefile=savefile)
        all_y.extend(valy)
        all_preds.extend(ypreds)

    return all_y, all_preds


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

    # def get_select_cols(self, cols):
    #     """
    #     If you happen to use a conf file that results in too much data
    #     and want to clean it up, select only the columns you want.
    #     suprafile: the path to a csv file containing results
    #     cols: an array of columns that you want to select
    #     Returns data as an np array
    #     """
    #     suprafile = "{0}/{1}.csv".format(self.apath, self.supra_name)
    #     supras = pd.read_csv(suprafile, sep=',')
    #     try:
    #         return supras[cols]
    #     except:
    #         for col in cols:
    #             if col not in supras.columns:
    #                 cols.remove(col)
    #         return supras[cols].to_numpy()


class PrepareData:
    """
    Instances of class PrepareData take segmental, suprasegmental, and
    raw data in numpy format and prepares it for input into a NN.
    """
    def __init__(self, segmental_data=None, suprasegmental_data=None, raw_data=None):
        self.segmental = segmental_data
        self.suprasegmental = suprasegmental_data
        self.raw = raw_data
        self.model_type = self.determine_model()

    def determine_model(self):
        """
        Determines the type of model to be used with this data
        """
        model_type = 'all'
        if self.segmental is None:
            if self.suprasegmental is None:
                model_type = 'raw'
            else:
                if self.raw is None:
                    model_type = 'suprasegmental'
                else:
                    model_type = 'supra_and_raw'
        elif self.suprasegmental is None:
            if self.raw is None:
                model_type = 'segmental'
            else:
                model_type = 'seg_and_raw'
        elif self.raw is None:
            model_type = 'seg_and_supra'
        return model_type

    def concat_data(self):
        """
        Concatenate the segmental and suprasegmental data
        todo: complete 'pass' statements for other model types
        todo: determine how to concat raw data
        """
        if self.model_type == 'seg_and_supra':
            return np.concatenate((self.segmental, self.suprasegmental), axis=1)
        elif self.model_type == 'seg_and_raw':
            return self.segmental, self.raw
        elif self.model_type == 'supra_and_raw':
            return self.suprasegmental, self.raw
        elif self.model_type == 'all':
            return np.concatenate((self.segmental, self.suprasegmental), axis=1), self.raw
        elif self.model_type == 'segmental':
            return self.segmental
        elif self.model_type == 'suprasegmental':
            return self.suprasegmental
        elif self.model_type == 'raw':
            return self.raw
        else:
            return warnings.warn("Something is wrong here...model_type note found")

    def get_data_size(self):
        return self.concat_data().size()


class AdaptiveModel:
    """
    Should allow for input of 1, 2, or all 3 types of data;
    todo: should all types be handled with the same architecture?
    Assumes that that x and y have been aligned
    Right now, this is agnostic to type of data--assumes concatenated elements
    todo: we want to have models for the following
            raw RNN, phonetic RNN, phonol MLP
            raw + phonet RNN, raw + phonol ?, phonet + phonol ?
            raw + phonet + phonol ?
            for '?' we can do voting/weighted avg of scores
               or we can do final layer of additional MLP
            timestamped features better for RNN (phonetic LLDs)
            could try this OR high level vector when combining
    """
    def __init__(self, xdata, ydata, data_shape, outpath):
        self.data = zip(xdata, ydata)
        self.xs = xdata
        self.ys = ydata
        self.data_shape = data_shape
        self.save_path = outpath
        # self.model = Sequential()

    def split_data_for_cv(self, k=10):
        """
        Split the data into k folds for CV
        k : number of folds
        Returns dicts of n : data and n : ys
        """
        cv_data_dict = {}
        cv_ys_dict = {}

        data = list(zip(self.xs, self.ys))
        random.shuffle(data)

        total_length = self.data_shape[0]

        k_length = round(total_length / k)

        for i in range(k-1):
            fold_data = data[i*k_length:(i+1)*k_length]
            fold_xs, fold_ys = list(zip(*fold_data))
            cv_data_dict[i+1] = np.array(fold_xs)
            cv_ys_dict[i+1] = np.array(fold_ys)
        final_fold_data = data[(k-1)*k_length:]
        final_xs, final_ys = list(zip(*final_fold_data))
        cv_data_dict[k] = np.array(final_xs)
        cv_ys_dict[k] = np.array(final_ys)

        return cv_data_dict, cv_ys_dict

    def split_data(self, train=0.7, test=0.15):
        """
        Split the data into training, dev and testing folds.
        The params specify proportion of data in train and dev.
        Dev data proportion is 1 - (train + test).
        Assumes data is zipped x & y
        """
        data = list(zip(self.xs, self.ys))
        random.shuffle(data)

        total_length = self.data_shape[0]

        train_length = round(total_length * train)
        test_length = round(total_length * test)

        xs, ys = list(zip(*self.data))

        trainX = np.array(xs[:train_length])
        trainy = np.array(ys[:train_length])
        testX = np.array(xs[train_length:train_length + test_length])
        testy = np.array(ys[train_length:train_length + test_length])
        valX = np.array(xs[train_length + test_length:])
        valy = np.array(ys[train_length + test_length:])

        return trainX, trainy, valX, valy, testX, testy

    def save_data(self, trainX, trainy, valX, valy, testX, testy):
        # save data folds created above
        np.save("{0}/X_train".format(self.save_path), trainX)
        np.save("{0}/y_train".format(self.save_path), trainy)
        np.save("{0}/X_val".format(self.save_path), valX)
        np.save("{0}/y_val".format(self.save_path), valy)
        np.save("{0}/X_test".format(self.save_path), testX)
        np.save("{0}/y_test".format(self.save_path), testy)

    def load_existing_data(self, train_X_file, train_y_file, val_X_file, val_y_file,
                           test_X_file, test_y_file):
        """
        Load existing files; all files should be in .npy format in the save path.
        """
        trainX = np.load("{0}/{1}".format(self.save_path, train_X_file))
        trainy = np.load("{0}/{1}".format(self.save_path, train_y_file))
        valX = np.load("{0}/{1}".format(self.save_path, val_X_file))
        valy = np.load("{0}/{1}".format(self.save_path, val_y_file))
        testX = np.load("{0}/{1}".format(self.save_path, test_X_file))
        testy = np.load("{0}/{1}".format(self.save_path, test_y_file))

        return trainX, trainy, valX, valy, testX, testy

    def lr_model(self, act='tanh', output_size=1, loss_fx="mean_squared_error",
                 l_rate=0.001, beta_1=0.9, beta_2=0.999):
        model = Sequential()
        model.add(Dense(output_dim=output_size, input_dim=self.data_shape[1:],
                        activation=act))
        # set an optimizer -- adam with default param values
        opt = optimizers.Adam(learning_rate=l_rate, beta_1=beta_1, beta_2=beta_2)
        model.compile(optimizer=opt, loss=loss_fx)
        return model

    def mlp_model(self, n_connected=2, n_connected_units=25, l_rate=0.001,
                  dropout=0.2, beta_1=0.9, beta_2=0.999, act='tanh',
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
        output_size:            the length of predictions vector; default is 7
        """
        model = Sequential()
        print(self.data_shape)
        print(self.data_shape[1:])
        if n_connected >= 2:
            while n_connected > 1:
                model.add(Dense(n_connected_units, input_dim=self.data_shape[-1],
                                     activation=act))
                model.add(Dropout(dropout))
                n_connected -= 1
            # add the final layer with output activation
            model.add(Dense(output_size, activation=output_act))
        else:
            # just feed it through linearly if for some reason n_connected = 1
            # this is no longer an mlp, though
            model.add(Dense(output_size, input_dim=self.data_shape[-1],
                            activation=output_act, dropout=dropout))
        # set an optimizer -- adam with default param values
        opt = optimizers.Adam(learning_rate=l_rate, beta_1=beta_1, beta_2=beta_2)
        # compile the model
        # model.compile(loss=loss_fx, optimizer=opt, metrics=['acc'])
        model.compile(loss=loss_fx, optimizer=opt, metrics=[r_squared])
        return model

    def lstm_model(self, n_lstm=2, n_lstm_units=50, dropout=0.2, n_connected=1,
                     n_connected_units=25, l_rate = 0.001, beta_1=0.9, beta_2=0.999,
                     act='relu', output_act='linear', loss_fx='mean_squared_error',
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
        if n_lstm > 1:
            while n_lstm > 1:
                model.add(Bidirectional(LSTM(n_lstm_units,
                                                  activation=act,
                                                  input_shape=self.data_shape[1:], dropout=dropout,
                                                  recurrent_dropout=dropout, return_sequences=True)))
                n_lstm -= 1
            # print("N LSTM layers left equals: " + str(n_lstm))
            model.add(Bidirectional(LSTM(n_lstm_units, input_shape=self.data_shape[1:],
                                              activation=act, dropout=dropout, recurrent_dropout=dropout,
                                              return_sequences=False)))
            n_lstm -= 1
            # print("THE LSTM layers completed")
        else:
            model.add(Bidirectional(LSTM(n_lstm_units, input_shape=self.data_shape[1:],
                                              activation=act, dropout=dropout, recurrent_dropout=dropout,
                                              return_sequences=False)))
        # add the connected layers
        while n_connected > 0:
            model.add(Dense(n_connected_units, input_shape=(self.data_shape[0],1),
                                 activation=act))
            n_connected -= 1
        # print("The connected layer worked")
        # add the final layer with output activation
        model.add(Dense(output_size, activation=output_act))
        # set an optimizer -- adam with default param values
        # print("The output layer worked")
        opt = optimizers.Adam(learning_rate=l_rate, beta_1=beta_1, beta_2=beta_2)
        # compile the model
        model.compile(loss=loss_fx, optimizer=opt, metrics=[r_squared])
        print("Model compiled successfully")
        return model

    def final_layers(self, n_connected=1, n_connected_units=25, l_rate=0.001,
                  dropout=0.2, beta_1=0.9, beta_2=0.999, act='relu',
                  output_act='linear', loss_fx='mean_squared_error'):
        """
        Initialize the MLP model as that takes output from lstm + mlp models and combines
        """
        # create an mlp layer
        # create an lstm layer
        # compile mlp and lstm
        # train the final layer model on the TRAINING output of each model
