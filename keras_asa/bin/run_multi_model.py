# run the code created in multi_input_model.py

import os, sys
import pprint
import numpy as np
import multi_input_model as modeler

# create an instance of GetFeatures class
phonetic_test = modeler.GetFeatures("../../SJP_JC_Audio/S07/wav", "~/opensmile-2.3.0", "../../SJP_JC_Audio/output")
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
# print(testX)

# uncomment the following lines to save and load folds
# adapt.save_data(trainX, trainy, valX, valy, testX, testy)
# trainX, trainy, valX, valy, testX, testy = adapt.load_existing_data("X_train.npy", "y_train.npy",
#                                                                     "X_val.npy", "y_val.npy",
#                                                                     "X_test.npy", "y_test.npy")

# try out lstm model
adapt.lstm_model()
adapt.train_and_predict(trainX, trainy, valX, valy, batch=5)
