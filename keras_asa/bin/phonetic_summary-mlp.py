# run the code created in multi_input_model.py
from statistics import mean

import numpy as np
import sys
from sklearn.metrics import mean_squared_error

import multi_input_model as modeler
import matplotlib.pyplot as plt


def calc_r_squared(valy, y_preds):
    mean_y = mean(valy)
    ss_res = []
    ss_tot = []
    for i, item in enumerate(valy):
        ss_res.append((item - y_preds[i]) ** 2)
        ss_tot.append((item - mean_y) ** 2)
    r_value = (1 - sum(ss_res) / (sum(ss_tot) + 0.0000001))
    return float(r_value)

#####################################################
########### RUN SUMMARY PHONETIC TEST ###############
# one set of input per data point -- NOT RNN; use MLP instead


dset_paths = ["../../SJP_JC_Audio/IS10_summary_test", "../../SJP_JC_Audio/IS09_summary_test"]

for dpath in dset_paths:
    # create an instance of GetFeatures class
    summary_test = modeler.GetFeatures("../../SJP_JC_Audio/wavs", "~/opensmile-2.3.0", dpath)
    # uncomment to extract features for first time use
    # summary_test.extract_features(summary_stats=True)

    # get dict with all features for the dataset
    summary_features = summary_test.get_features_dict()

    # select participants to use
    speaker_list = ["S02", "S03", "S04", "S05", "S07", "S08", "S09", "S19", "S21", "S22", "S23", "S24", "S25", "S26", "S28"]

    types = ["../../SJP_JC_Audio/perception_results/accented_avgs.csv",
             "../../SJP_JC_Audio/perception_results/fluency_avgs.csv",
             "../../SJP_JC_Audio/perception_results/comp_avgs.csv"]

    for fpath in types:
        # read in y data
        summary_y_values = modeler.get_ys_dict(fpath, speaker_list)

        # zip x and y data
        summary_zipped = modeler.zip_feats_and_ys(summary_features, summary_y_values)

        # test to make sure this worked
        unzipped_summary_feats, unzipped_summary_ys = zip(*summary_zipped)

        # set variables for input into model
        unzipped_summary_ys = list(unzipped_summary_ys)
        unzipped_summary_feats = np.array(list(unzipped_summary_feats))
        summary_shape = unzipped_summary_feats.shape

        # create instance of class AdaptiveModel
        sum_adapt = modeler.AdaptiveModel(unzipped_summary_feats, unzipped_summary_ys, summary_shape, dpath)

        # split data into datasets
        # todo: this is icky but should work
        cv_data, cv_ys = sum_adapt.split_data_for_cv(k=10)

        cv_updated = {}

        for k, v in cv_data.items():
            cv_updated[k] = modeler.reshape_data(v)
        cv_data = cv_updated

        # hyperparameter tuning on MLP model
        n_connected_units = [5, 10, 20, 50, 100, 200, 400] # higher than 5 is better...
        n_connected = [2, 3, 4, 5, 10] # larger numbers fare worse
        batch = [12, 32, 64, 100]
        learning_rate = [0.001, 0.005, 0.01, 0.1]

        total_stats = [['mse', 'r_squared', 'number_connected_units', 'number_connected_layers', 'batch_size',
                        'learning_rate']]

        # moderate-sized grid search
        for n_c_u in n_connected_units:
            for  n_c in n_connected:
                for b in batch:
                    for l_r in learning_rate:

                        print("Number of connected units: " + str(n_c_u))
                        print("Number of connected layers: " + str(n_c))
                        print("Batch size: " + str(b))
                        print("Learning rate: " + str(l_r))

                        sum_model = sum_adapt.mlp_model(n_connected=n_c, n_connected_units=n_c_u, act='tanh', output_act='linear',
                                                        output_size=1)

                        valy, y_preds = modeler.cv_train_wrapper(sum_model, cv_data, cv_ys, batch=b, num_epochs=100)

                        r_value = calc_r_squared(valy, y_preds)
                        mse = mean_squared_error(valy, y_preds)

                        model_stats = [str(mse), str(r_value), str(n_c_u), str(n_c), str(b), str(l_r)]
                        total_stats.append(model_stats)

        if fpath == "../../SJP_JC_Audio/perception_results/comp_avgs.csv":
            if dpath == "../../SJP_JC_Audio/IS10_summary_test":
                with open('../../SJP_JC_Audio/gridsearch/phonetic_summaryIS10_mlp_comprehensibility.csv', 'w') as wfile:
                    for item in total_stats:
                        wfile.write(",".join(item))
                        wfile.write("\n")
            else:
                with open('../../SJP_JC_Audio/gridsearch/phonetic_summaryIS09_mlp_comprehensibility.csv', 'w') as wfile:
                    for item in total_stats:
                        wfile.write(",".join(item))
                        wfile.write("\n")
        elif fpath == "../../SJP_JC_Audio/perception_results/fluency_avgs.csv":
            if dpath == "../../SJP_JC_Audio/IS10_summary_test":
                with open('../../SJP_JC_Audio/gridsearch/phonetic_summaryIS10_mlp_fluency.csv', 'w') as wfile:
                    for item in total_stats:
                        wfile.write(",".join(item))
                        wfile.write("\n")
            else:
                with open('../../SJP_JC_Audio/gridsearch/phonetic_summaryIS09_mlp_fluency.csv', 'w') as wfile:
                    for item in total_stats:
                        wfile.write(",".join(item))
                        wfile.write("\n")
        else:
            if dpath == "../../SJP_JC_Audio/IS10_summary_test":
                with open('../../SJP_JC_Audio/gridsearch/phonetic_summaryIS10_mlp_accentedness.csv', 'w') as wfile:
                    for item in total_stats:
                        wfile.write(",".join(item))
                        wfile.write("\n")
            else:
                with open('../../SJP_JC_Audio/gridsearch/phonetic_summaryIS09_mlp_accentedness.csv', 'w') as wfile:
                    for item in total_stats:
                        wfile.write(",".join(item))
                        wfile.write("\n")