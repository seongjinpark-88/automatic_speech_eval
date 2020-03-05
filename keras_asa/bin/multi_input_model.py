# prepare model that accepts multiple types of input

import os
import numpy as np
import pandas as pd

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


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
    """
    def __init__(self, data, data_size):
        self.data = data
        self.data_size = data_size
        self.model = Sequential()

    def define_model(self, n_hidden=1, act='relu', output_act='softmax'):
        """
        Initialize the model
        n_hidden: the number of hidden layers
        """
        self.model.add(Dense(25, input_dim=self.data_size[1], activation=act))
        self.model.add(Dense(7, activation=output_act))