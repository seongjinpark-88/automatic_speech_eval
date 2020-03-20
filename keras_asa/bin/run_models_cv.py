# run the code created in multi_input_model.py for CV
from collections import OrderedDict

import numpy as np
import multi_input_model as modeler
import sys
import matplotlib.pyplot as plt


# get phonological data
phono_feats = modeler.get_phonological_features("../../SJP_JC_Audio/rhythm.csv")

# select participants to use
speaker_list = ["S02", "S03", "S04", "S05", "S07", "S08", "S09", "S19", "S21", "S22", "S23", "S24", "S25", "S26", "S28"]

# read in y data
summary_y_values = modeler.get_ys_dict("../../SJP_JC_Audio/perception_results/accented_avgs.csv",
                                    speaker_list)

# zip x and y data
summary_zipped = modeler.zip_feats_and_ys(phono_feats, summary_y_values)

# test to make sure this worked
unzipped_summary_feats, unzipped_summary_ys = zip(*summary_zipped)

# set variables for input into model
unzipped_summary_ys = list(unzipped_summary_ys)
unzipped_summary_feats = np.array(list(unzipped_summary_feats))
summary_shape = unzipped_summary_feats.shape

# create instance of class AdaptiveModel
sum_adapt = modeler.AdaptiveModel(unzipped_summary_feats, unzipped_summary_ys, summary_shape,
                                  "../../SJP_JC_Audio/phonological_test")

# split data into datasets
cv_data, cv_ys = sum_adapt.split_data_for_cv(k=10)


# hyperparameter tuning on MLP model
n_connected_units = [5, 10, 20, 50, 100, 200, 400] # higher than 5 is better...
n_connected = [2, 3, 4, 5, 10] # larger numbers fare worse
batch = [12, 32, 64, 100]

# moderate-sized grid search
for n_c_u in n_connected_units:
    for n_c in n_connected:
        for b in batch:

            print("Number of connected units: " + str(n_c_u))
            print("Number of connected layers: " + str(n_c))
            print("Batch size: " + str(b))

            sum_model = sum_adapt.mlp_model(n_connected=n_c, n_connected_units=n_c_u, act='tanh', output_act='linear',
                                        output_size=1)

            valy, y_preds = modeler.cv_train_wrapper(sum_model, cv_data, cv_ys, batch=b, num_epochs=1000)

            plt.plot([0, 6], [0, 6], 'k-', color='red')
            plt.scatter(valy, y_preds, color='blue', label='data')

            plt.legend()
            plt.xlabel('observed (y)')
            plt.xlim(0, 6)
            plt.ylabel('predicted (y-hat)')
            plt.ylim(0, 6)

            plt.title('y vs y-hat {0} connected units, {1} connected layers, batch size {2}'.format(n_c_u, n_c, b))
            plt.savefig("phonological_dev-loss_{0}-{1}-{2}.png".format(n_c_u, n_c, b))
            plt.clf()
