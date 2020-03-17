# run the code created in multi_input_model.py

import os, sys
import pprint
import numpy as np
from keras import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense

import multi_input_model as modeler

# create an instance of GetFeatures class
phonetic_test = modeler.GetFeatures("../../SJP_JC_Audio/wavs", "~/opensmile-2.3.0", "../../SJP_JC_Audio/output")
# phonetic_test.copy_files_to_single_directory("../../SJP_JC_Audio/all_speakers")
acoustic_features = phonetic_test.get_features_dict(supra=False)

# read in y data
y_values = phonetic_test.get_ys_dict("../../SJP_JC_Audio/perception_results/accented_avgs.csv")

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
# trainX, trainy, valX, valy, testX, testy = adapt.load_existing_data("X_train.npy", "y_train.npy",
#                                                                     "X_val.npy", "y_val.npy",
#                                                                     "X_test.npy", "y_test.npy")

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

adapt.lstm_model(output_size=1)

# adapt.model.fit(trainX, trainy, valX, valy)
# adapt.mode.summary()
#
# sys.exit(1)

adapt.train_and_predict(trainX, trainy, valX, valy, batch=5)
adapt.model.summary()