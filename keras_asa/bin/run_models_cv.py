# run the code created in multi_input_model.py for CV
from collections import OrderedDict

import numpy as np
import multi_input_model as modeler
import sys
from statistics import mean
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

dset_paths = ["../../SJP_JC_Audio/IS10_summary_test", "../../SJP_JC_Audio/IS09_summary_test"]

for dpath in dset_paths:
    # get phonological data
    phono_feats = modeler.get_phonological_features("../../SJP_JC_Audio/rhythm.csv")

    # get summary phonetic data, if using
    summary_feats = modeler.GetFeatures("../../SJP_JC_Audio/wavs", "~/opensmile-2.3.0", dpath)
    phonetic_feats = summary_feats.get_features_dict()

    combined_feats = modeler.combine_feat_types(phonetic_feats, phono_feats)

    # select participants to use
    speaker_list = ["S02", "S03", "S04", "S05", "S07", "S08", "S09", "S19", "S21",
                    "S22", "S23", "S24", "S25", "S26", "S28"]

    types = ["../../SJP_JC_Audio/perception_results/accented_avgs.csv",
             "../../SJP_JC_Audio/perception_results/fluency_avgs.csv",
             "../../SJP_JC_Audio/perception_results/comp_avgs.csv"]

    for fpath in types:
        # read in y data
        # if getting accentedness predictions
        # summary_y_values = modeler.get_ys_dict("../../SJP_JC_Audio/perception_results/accented_avgs.csv",
        #                                     speaker_list)

        # if getting fluency predictions
        # summary_y_values = modeler.get_ys_dict("../../SJP_JC_Audio/perception_results/fluency_avgs.csv",
        #                                        speaker_list)

        # if getting comprehensibility predictions
        # summary_y_values = modeler.get_ys_dict("../../SJP_JC_Audio/perception_results/comp_avgs.csv",
        #                                        speaker_list)

        summary_y_values = modeler.get_ys_dict(fpath, speaker_list)

        # zip x and y data
        # summary_zipped = modeler.zip_feats_and_ys(phono_feats, summary_y_values)
        summary_zipped = modeler.zip_feats_and_ys(combined_feats, summary_y_values)

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
        # n_connected_units = [400]
        # n_connected = [2]
        # batch = [100]
        n_connected_units = [10, 20, 50, 100, 200, 400] # higher than 5 is better...
        n_connected = [2, 3, 5, 10]  # larger numbers fare worse
        batch = [12, 32, 64, 100]
        learning_rates = [0.0005, 0.001, 0.005, 0.01]

        total_stats = [['mse', 'r_squared', 'number_connected_units', 'number_connected_layers', 'batch_size', 'learning_rate']]
        # moderate-sized grid search
        for n_c_u in n_connected_units:
            for n_c in n_connected:
                for b in batch:
                    for l_r in learning_rates:

                        print("Number of connected units: " + str(n_c_u))
                        print("Number of connected layers: " + str(n_c))
                        print("Batch size: " + str(b))
                        print("Learning rate: " + str(l_r))

                        sum_model = sum_adapt.mlp_model(n_connected=n_c, n_connected_units=n_c_u, l_rate=l_r,
                                                        act='tanh', output_act='linear',
                                                        output_size=1)

                        valy, y_preds = modeler.cv_train_wrapper(sum_model, cv_data, cv_ys, batch=b, num_epochs=1000) # ,
                                                                 # savefile="fluency_phonological_cv_{0}-{1}-{2}.csv".format(n_c_u,
                                                                 #                                                           n_c, b))

                        mean_y = mean(valy)
                        ss_res = []
                        ss_tot = []
                        for i, item in enumerate(valy):
                            ss_res.append((item - y_preds[i]) ** 2)
                            ss_tot.append((item - mean_y) ** 2)
                        r_value = (1 - sum(ss_res) / (sum(ss_tot) + 0.0000001))
                        r_value = float(r_value)
                        # r_value = modeler.r_squared(np.asarray(valy), np.asarray(y_preds))
                        # print(r_value)

                        mse = mean_squared_error(valy, y_preds)

                        # plt.plot([0, 6], [0, 6], 'k-', color='red')
                        # plt.scatter(valy, y_preds, color='blue', label='data')
                        #
                        # plt.legend()
                        # plt.xlabel('observed (y)')
                        # plt.xlim(0, 6)
                        # plt.ylabel('predicted (y-hat)')
                        # plt.ylim(0, 6)
                        # # plt.rcParams["axes.titlesize"] = 8
                        #
                        # plt.title('y vs y-hat {0} connected units, {1} connected layers, batch size {2},\nmse {3}, r_squared {4}'.format(n_c_u, n_c, b, mse, r_value))
                        # plt.savefig("{0}_{1}_fluency_phonological_cv_{2}-{3}-{4}.png".format(r_value, mse, n_c_u, n_c, b))
                        # plt.clf()

                        model_stats = [str(mse), str(r_value), str(n_c_u), str(n_c), str(b), str(l_r)]
                        total_stats.append(model_stats)

        if fpath == "../../SJP_JC_Audio/perception_results/comp_avgs.csv":
            if dpath == "../../SJP_JC_Audio/IS10_summary_test":
                with open('../../SJP_JC_Audio/gridsearch/phono+phonetic_summaryIS10_mlp_comprehensibility.csv', 'a+') as wfile:
                    for item in total_stats:
                        wfile.write(",".join(item))
                        wfile.write("\n")
            else:
                with open('../../SJP_JC_Audio/gridsearch/phono+phonetic_summaryIS09_mlp_comprehensibility.csv', 'a+') as wfile:
                    for item in total_stats:
                        wfile.write(",".join(item))
                        wfile.write("\n")
        elif fpath == "../../SJP_JC_Audio/perception_results/fluency_avgs.csv":
            if dpath == "../../SJP_JC_Audio/IS10_summary_test":
                with open('../../SJP_JC_Audio/gridsearch/phono+phonetic_summaryIS10_mlp_fluency.csv', 'a+') as wfile:
                    for item in total_stats:
                        wfile.write(",".join(item))
                        wfile.write("\n")
            else:
                with open('../../SJP_JC_Audio/gridsearch/phono+phonetic_summaryIS09_mlp_fluency.csv', 'a+') as wfile:
                    for item in total_stats:
                        wfile.write(",".join(item))
                        wfile.write("\n")
        else:
            if dpath == "../../SJP_JC_Audio/IS10_summary_test":
                with open('../../SJP_JC_Audio/gridsearch/phono+phonetic_summaryIS10_mlp_accentedness.csv', 'a+') as wfile:
                    for item in total_stats:
                        wfile.write(",".join(item))
                        wfile.write("\n")
            else:
                with open('../../SJP_JC_Audio/gridsearch/phono+phonetic_summaryIS09_mlp_accentedness.csv', 'a+') as wfile:
                    for item in total_stats:
                        wfile.write(",".join(item))
                        wfile.write("\n")
