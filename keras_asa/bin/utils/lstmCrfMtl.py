import numpy as np

import random

seed = 888
random.seed(seed)
np.random.seed(seed)

import tensorflow as tf

import keras
from keras import backend as K
from keras import optimizers

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.layers import LSTM, Bidirectional


def r_squared(y_true, y_pred):
    """
    r-squared calculation
    from: https://jmlb.github.io/ml/2017/03/20/CoeffDetermination_CustomMetric4Keras/
    """
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - (ss_res / (ss_tot + K.epsilon()))


class lstmMtl:
    def __init__(self, X, name, model_type):
        """
        @param X: input features
        @param name: name of the model
        @param model_type: type of model (lstm or mlp)
        """
        if model_type == "lstm":
            self.input_shape = (np.shape(X)[1], np.shape(X)[2])
        elif model_type == "mlp":
            self.input_shape = (np.shape(X)[-1],)
        else:
            print("Undefined input shape")
            exit()

        self.name = name

    def bi_lstm_model(self, n_lstm_units=512, dropout=0.2, n_connected_units=32, act='tanh', l_rate=0.001, loss_fx='mse'):
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

        self.lstm_input = Input(shape=self.input_shape, name=self.name)

        output_1 = Bidirectional(LSTM(n_lstm_units, activation=act,
                                      dropout=dropout, recurrent_dropout=dropout, return_sequences=True))(
            self.lstm_input)
        output_2 = Bidirectional(LSTM(n_lstm_units, activation=act,
                                      dropout=dropout, recurrent_dropout=dropout, return_sequences=False))(output_1)

        output_3 = Dense(n_connected_units, activation='tanh')(output_2)
        dropout_3 = Dropout(dropout)(output_3)
        output_4 = Dense(256, activation='relu')(dropout_3)
        dropout_4 = Dropout(dropout)(output_4)

        self.acc_output = Dense(1, activation='linear', name='acc_output')(dropout_4)
        self.flu_output = Dense(1, activation='linear', name='flu_output')(dropout_4)
        self.com_output = Dense(1, activation='linear', name='com_output')(dropout_4)

        self.model = Model(inputs=self.lstm_input, outputs=[self.acc_output, self.flu_output, self.com_output])

        adam = optimizers.Adam(learning_rate=l_rate)

        self.model.compile(optimizer=adam, loss=[loss_fx, loss_fx, loss_fx], metrics={'acc_output': loss_fx})
        self.model.summary()

        # return self.model

    def train_model(self, epochs=100, batch_size=64, input_feature=None,
                    output_label=None, validation=None, model_name=None):

        # early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=10)
        early_stopping = EarlyStopping(monitor='val_acc_output_loss', mode='min', patience=10)
        # model_name = model_name + ".h5"

        # save_best = ModelCheckpoint(model_name, monitor='val_loss', mode='min')

        self.history = self.model.fit(input_feature, output_label,
                                      epochs=epochs, batch_size=batch_size, shuffle=True,
                                      # validation_data = validation, verbose = 1,
                                      validation_split=0.1, verbose=1,
                                      callbacks=[early_stopping])
        return self.history

    def predict_model(self, input_feature=None, prediction_type=0):
        self.pred = self.model.predict(input_feature)[prediction_type]
        return self.pred

    def evaluate_model(self, input_feature, true_label):
        self.scores = self.model.evaluate(input_feature, true_label, verbose=0)
        return self.scores
