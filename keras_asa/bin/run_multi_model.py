# run the code created in multi_input_model.py

import os
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
unzipped_feats = np.array(list(unzipped_feats))
shape = unzipped_feats.shape
#print(type(unzipped_feats))
#print(unzipped_feats.shape)
#pprint.pprint(unzipped_feats[0])
# success!

# create instance of class AdaptiveModel
adapt = modeler.AdaptiveModel(zipped, shape, "../../SJP_JC_Audio/output")

# split data into datasets
trainX, trainy, valX, valy, testX, testy = adapt.split_data()
print(testX)