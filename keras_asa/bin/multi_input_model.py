# prepare model that accepts multiple types of input

import os
import numpy as np
import pandas as pd
import pickle
import random
import pprint

# set seed for reproducibility
seed = 888
random.seed(seed)
np.random.seed(seed)

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Bidirectional, Dropout, Flatten
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import classification_report

#todo: finish data prep code, scaffold NN code in class AdaptiveModel
#todo: ensure that we can actually get the feature sets we want -- work on this


class GetFeatures:
    """
    Takes input files and gets segmental and/or suprasegmental features
    Current features extracted: XXXX, YYYY, ZZZZ
    """
    def __init__(self, audio_path, opensmile_path):
        self.apath = audio_path
        self.smilepath = opensmile_path
        self.supra_name = None
        self.segment_name = None

    def get_features(self, output_name, supra=True):
        for f in os.listdir(self.apath):
            if f.endswith('.wav'):
                # todo: replace config files with the appropriate choice
                if supra is True:
                    os.system("{0}/SMILExtract -C {0}/config/IS10_paraling.conf -I {1}/{2}\
                          -csvoutput {3}".format(self.smilepath, self.apath, f, output_name))
                    self.segment_name = output_name
                else:
                    os.system("{0}/SMILExtract -C {0}/config/IS09_paraling.conf -I {1}/{2}\
                          -csvoutput {3}".format(self.smilepath, self.apath, f, output_name))
                    self.supra_name = output_name

    def get_select_cols(self, cols):
        """
        If you happen to use a conf file that results in too much data
        and want to clean it up, select only the columns you want.
        suprafile: the path to a csv file containing results
        cols: an array of columns that you want to select
        Returns data as an np array
        """
        suprafile = "{0}/{1}.csv".format(self.apath, self.supra_name)
        supras = pd.read_csv(suprafile, sep=',')
        try:
            return supras[cols]
        except:
            for col in cols:
                if col not in supras.columns:
                    cols.remove(col)
            return supras[cols].to_numpy()


class PrepareData:
    """
    Instances of class PrepareData take segmental, suprasegmental, and
    raw data in numpy format and prepares it for input into a NN.
    """
    def __init__(self, segmental_data=None, suprasegmental_data=None, raw_data=None):
        self.segmental = segmental_data
        self.suprasegmental = suprasegmental_data
        self.raw = raw_data

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
        """
        if self.segmental is not None and self.suprasegmental is not None:
            return np.concatenate((self.segmental, self.suprasegmental), axis=1)

    def get_data_size(self):
        return self.concat_data().size()


class AdaptiveModel:
    """
    Should allow for input of 1, 2, or all 3 types of data;
    todo: should all types be handled with the same architecture?
    Assumes that the y for each data point is the FINAL ELEMENT of the vector
    """
    def __init__(self, data, data_size, outpath):
        self.data = data
        self.data_size = data_size
        self.save_path = outpath
        self.model = Sequential()

    def split_data(self, train=0.7, dev=0.15):
        """
        Split the data into training, dev and testing folds.
        The params specify proportion of data in train and dev.
        Test data proportion is 1 - (train + dev).
        """
        np.random.permutation(self.data)
        total_length = self.data_size[0]
        train_length = round(total_length * train)
        dev_length = round(total_length * dev)

        trainset = self.data[:train_length]
        devset = self.data[train_length:dev_length]
        testset = self.data[dev_length:]

        trainX = [item[:-1] for item in trainset]
        trainy = [item[-1] for item in trainset]
        valX = [item[:-1] for item in devset]
        valy = [item[-1] for item in devset]
        testX = [item[:-1] for item in testset]
        testy = [item[-1] for item in testset]

        return trainX, trainy, valX, valy, testX, testy

    def save_data(self, trainX, trainy, valX, valy, testX, testy):
        # save data folds created above
        pickle.dump(trainX, open("X_train.h5", 'w'))
        pickle.dump(trainy, open("y_train.h5", 'w'))
        pickle.dump(valX, open("X_val.h5", 'w'))
        pickle.dump(valy, open("y_val.h5", 'w'))
        pickle.dump(testX, open("X_test.h5", 'w'))
        pickle.dump(testy, open("y_test.h5", 'w'))

    def load_existing_data(self, train_X_file, train_y_file, val_X_file, val_y_file,
                           test_X_file, test_y_file):
        """
        Load existing files; all files should be in h5 format in the save path.
        """
        trainX = pickle.load("{0}/{1}".format(self.save_path, train_X_file))
        trainy = pickle.load("{0}/{1}".format(self.save_path, train_y_file))
        valX = pickle.load("{0}/{1}".format(self.save_path, val_X_file))
        valy = pickle.load("{0}/{1}".format(self.save_path, val_y_file))
        testX = pickle.load("{0}/{1}".format(self.save_path, test_X_file))
        testy = pickle.load("{0}/{1}".format(self.save_path, test_y_file))
        return trainX, trainy, valX, valy, testX, testy

    def define_model(self, n_lstm=2, n_lstm_units=50, dropout=0.2, n_connected=1,
                     n_connected_units=25, l_rate = 0.001, beta_1=0.9, beta_2=0.999,
                     act='relu', output_act='softmax',loss_fx='mean_squared_error'):
        """
        Initialize the model
        n_lstm:                 number of lstm layers
        n_lstm_units:           number of lstm cells in each layer
        dropout:                dropout rate in lstm layers
        n_connected:            the number of fully connected layers
        n_connected_units:      number of cells in connected layers
        beta_1:                 value of beta 1 for Adam
        beta_2:                 value of beta 2 for Adam
        act:                    the activation function in lstm + dense layers
        output_act:             the activation function in the final layer
        """
        # add all the hidden layers
        while n_lstm > 0:
            self.model.add(LSTM(n_lstm_units, input_dim=self.data_size[1],
                                activation=act, dropout=dropout))
            n_lstm -= 1
        # add the connected layers
        while n_connected > 0:
            self.model.add(Dense(n_connected_units, input_dim=self.data_size[1],
                                 activation=act))
            n_connected -= 1
        # add the final layer with output activation
        self.model.add(Dense(7, activation=output_act))
        # set an optimiser -- adam with default param values
        opt = optimizers.Adam(learning_rate=l_rate, beta_1=beta_1, beta_2=beta_2)
        # compile the model
        self.model.compile(loss=loss_fx, optimizer=opt, metrics=['acc'])

    def train_and_predict(self, trainX, trainy, valX, valy, batch=32, num_epochs=100):
        """
        Train the model
        batch: minibatch size
        """
        # fit the model to the data
        self.model.fit(trainX, trainy, batch_size=batch, epochs=num_epochs, shuffle=True)
        # get predictions on the dev set
        y_preds = self.model.predict(valX, batch_size=batch)
        pprint.pprint(classification_report(valy, y_preds))

    def save_model(self, m_name='best_model.h5'):
        self.model.save(m_name)