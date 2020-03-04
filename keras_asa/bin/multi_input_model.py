# prepare model that accepts multiple types of input

import os
import numpy as np
import pandas as pd


def get_select_cols(suprafile, cols):
    """
    If you happen to use a conf file that results in too much data
    and want to clean it up, select only the columns you want.
    suprafile: the path to a csv file containing results
    cols: an array of columns that you want to select
    """
    supras = pd.read_csv(suprafile, sep=',')
    try:
        return supras[cols]
    except:
        for col in cols:
            if col not in supras.columns:
                cols.remove(col)
        return supras[cols]


class Suprasegmentals:
    """
    Takes input files and gets suprasegmental features
    Current features extracted: XXXX, YYYY, ZZZZ
    """
    def __init__(self, audio_path, opensmile_path):
        self.apath = audio_path
        self.smilepath = opensmile_path

    def get_features(self, output_name):
        for f in os.listdir(self.apath):
            if f.endswith('.wav'):
                # todo: replace config file with the appropriate choice
                os.system("{0}/SMILExtract -C {0}/config/IS10_paraling.conf -I {1}\
                          -csvoutput {2}".format(self.smilepath, f, output_name))


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
            np.concatenate((self.segmental, self.suprasegmental), axis=1)
