# run the code created in multi_input_model.py

import os, sys
import pprint
import numpy as np
from keras import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
from keras.layers.advanced_activations import LeakyReLU

import multi_input_model as modeler

# create an instance of GetFeatures class
phonetic_test = modeler.GetFeatures("../../SJP_JC_Audio/wavs", "~/opensmile-2.3.0", "../../SJP_JC_Audio/output")
# phonetic_test.copy_files_to_single_directory("../../SJP_JC_Audio/all_speakers")
# phonetic_test.extract_features(supra=False)
acoustic_features = phonetic_test.get_features_dict()

# select participants to use
# speaker_list = ["S02", "S03", "S04", "S05", "S07", "S08", "S09", "S19", "S21", "S22", "S23", "S24", "S25", "S26", "S28"]
speaker_list = ["S07"]  # for testing
# read in y data
y_values = phonetic_test.get_ys_dict("../../SJP_JC_Audio/perception_results/accented_avgs.csv",
                                     speaker_list)

# zip x and y data
zipped = phonetic_test.zip_feats_and_ys(acoustic_features, y_values)

# test to make sure this worked
unzipped_feats, unzipped_ys = zip(*zipped)
#unzipped_feats = list(unzipped_feats)
#print(len(unzipped_feats), len(unzipped_ys))
#pprint.pprint(unzipped_feats)
unzipped_ys = list(unzipped_ys)
unzipped_feats = np.array(list(unzipped_feats))
shape = unzipped_feats.shape
#print(type(unzipped_feats))
#print(unzipped_feats.shape)
#pprint.pprint(unzipped_feats[0])
# success!

# create instance of class AdaptiveModel
adapt = modeler.AdaptiveModel(unzipped_feats, unzipped_ys, shape, "../../SJP_JC_Audio/output")

# print(adapt.data)
# print(len(adapt.xs))
# print(adapt.xs[0])
# print(adapt.ys)

# split data into datasets
trainX, trainy, valX, valy, testX, testy = adapt.split_data()
# pprint.pprint(testX[0])
# print(testX[0].shape)
# sys.exit(1)

# uncomment the following lines to save and load folds
# adapt.save_data(trainX, trainy, valX, valy, testX, testy)
trainX, trainy, valX, valy, testX, testy = adapt.load_existing_data("X_train.npy", "y_train.npy",
                                                                    "X_val.npy", "y_val.npy",
                                                                    "X_test.npy", "y_test.npy")

# try out lstm model

max_features = trainX.shape[2]
print(max_features)
maxlen = trainX.shape[1]
print(maxlen)
# print(trainX.shape)
# print(trainy.shape)
# sys.exit(1)

##### TRYING SOMETHING
# model = Sequential()
# model.add(Embedding(max_features, 128, input_length=maxlen))
# model.add(Bidirectional(LSTM(64)))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))
#
# # try using different optimizers and different optimizer configs
# model.compile('adam', 'mean_squared_error', metrics=['accuracy'])
#
# print('Train...')
# model.fit(trainX, trainy,
#           batch_size=5,
#           epochs=4,
#           validation_data=[testX, testy])
#
# model.summary()
# sys.exit(1)

adapt.lstm_model(n_lstm=1, output_size=1, n_lstm_units=10, n_connected_units=5,
                 act="tanh")  # relu doesn't work with this dataset

# adapt.model.fit(trainX, trainy, valX, valy)
# adapt.mode.summary()
#
# sys.exit(1)

valy, y_preds = adapt.train_and_predict(trainX, trainy, valX, valy, batch=12)

# print the linear regression and display datapoints
# from https://github.com/keras-team/keras/issues/7947
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

regressor = LinearRegression()
regressor.fit(valy.reshape(-1,1), y_preds)
y_fit = regressor.predict(y_preds)

reg_intercept = round(regressor.intercept_[0],4)
reg_coef = round(regressor.coef_.flatten()[0],4)
reg_label = "y = " + str(reg_intercept) + "*x +" + str(reg_coef)

plt.scatter(valy, y_preds, color='blue', label= 'data')
plt.plot(y_preds, y_fit, color='red', linewidth=2, label = 'Linear regression\n'+reg_label)
plt.title('Linear Regression')
plt.legend()
plt.xlabel('observed')
plt.ylabel('predicted')
plt.show()

#adapt.model.summary()