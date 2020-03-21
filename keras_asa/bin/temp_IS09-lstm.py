# run the code created in multi_input_model.py
from statistics import mean

import numpy as np
from sklearn.metrics import mean_squared_error

import multi_input_model as modeler

def calc_r_squared(valy, y_preds):
    mean_y = mean(valy)
    ss_res = []
    ss_tot = []
    for i, item in enumerate(valy):
        ss_res.append((item - y_preds[i]) ** 2)
        ss_tot.append((item - mean_y) ** 2)
    r_value = (1 - sum(ss_res) / (sum(ss_tot) + 0.0000001))
    return float(r_value)

######################################################
############ RUN INITIAL PHONETIC TEST ###############


# create an instance of GetFeatures class
phonetic_test = modeler.GetFeatures("../../SJP_JC_Audio/wavs", "~/opensmile-2.3.0", "../../SJP_JC_Audio/IS09_featureset")
# uncomment to extract features for first time use
# phonetic_test.extract_features()

# selected columns in IS09
acoustic_features = phonetic_test.get_features_dict()

# select participants to use
speaker_list = ["S02", "S03", "S04", "S05", "S07", "S08", "S09", "S19", "S21", "S22", "S23", "S24", "S25", "S26", "S28"]


types = ["../../SJP_JC_Audio/perception_results/accented_avgs.csv",
         "../../SJP_JC_Audio/perception_results/fluency_avgs.csv",
         "../../SJP_JC_Audio/perception_results/comp_avgs.csv"]

for fpath in types:
    # read in y data
    # accentedness
    y_values = modeler.get_ys_dict(fpath, speaker_list)

    # zip x and y data
    zipped = modeler.zip_feats_and_ys(acoustic_features, y_values)

    # test to make sure this worked
    unzipped_feats, unzipped_ys = zip(*zipped)

    # set variables for input into model
    unzipped_ys = list(unzipped_ys)
    unzipped_feats = np.array(list(unzipped_feats))
    shape = unzipped_feats.shape

    # create instance of class AdaptiveModel
    adapt = modeler.AdaptiveModel(unzipped_feats, unzipped_ys, shape, "../../SJP_JC_Audio/IS09_featureset")

    # split data into datasets
    cv_data, cv_ys = adapt.split_data_for_cv(k=10)

    # hyperparameter tuning on LSTM model
    n_lstm_units = [5, 10, 20, 50]
    n_connected_units = [5, 10, 20]
    n_lstm = [1, 2, 5]
    batch = [12, 64, 100]
    learning_rate = [0.001, 0.005, 0.01]

    # total_stats = [['mse', 'r_squared', 'number_lstm_units', 'number_lstm_layers', 'number_connected_units',
    #                 'batch_size', 'learning_rate']]

    if fpath == "../../SJP_JC_Audio/perception_results/fluency_avgs.csv":
        wfile = open('../../SJP_JC_Audio/gridsearch/phonetic_IS09-full_lstm_fluency.csv', 'a+')
    elif fpath == "../../SJP_JC_Audio/perception_results/comp_avgs.csv":
        wfile = open('../../SJP_JC_Audio/gridsearch/phonetic_IS09-full_lstm_comprehensibility.csv', 'a+')
    else:
        wfile = open('../../SJP_JC_Audio/gridsearch/phonetic_IS09-full_lstm_accentedness.csv', 'a+')

    wfile.write("mse,r_squared,number_lstm_units,number_lstm_layers,number_connected_units,batch_size,learning_rate\n")
    wfile.flush()

    # moderate-sized grid search
    for n_l_u in n_lstm_units:
        for n_c_u in n_connected_units:
            for n_l in n_lstm:
                for b in batch:
                    for l_r in learning_rate:
                        print("Number of lstm units: " + str(n_l_u))
                        print("Number of connected units: " + str(n_c_u))
                        print("Number of lstm layers: " + str(n_l))
                        print("Batch size: " + str(b))
                        print("Learning rate: " + str(l_r))

                        model = adapt.lstm_model(n_lstm=n_l, output_size=1, l_rate=l_r, n_lstm_units=n_l_u,
                                                 n_connected_units=n_c_u, act="tanh")

                        valy, y_preds = modeler.cv_train_wrapper(model, cv_data, cv_ys, batch=b, num_epochs=100)

                        r_value = calc_r_squared(valy, y_preds)
                        mse = mean_squared_error(valy, y_preds)


                        model_stats = [str(mse), str(r_value), str(n_l_u), str(n_l), str(n_c_u), str(b), str(l_r)]

                        wfile.write(",".join(model_stats))
                        wfile.write("\n")
                        wfile.flush()
                        # total_stats.append(model_stats)

    wfile.close()
    # if fpath == "../../SJP_JC_Audio/perception_results/comp_avgs.csv":
    #     with open('../../SJP_JC_Audio/gridsearch/phonetic_IS09-full_lstm_comprehensibility.csv', 'w') as wfile:
    #         for item in total_stats:
    #             wfile.write(",".join(item))
    #             wfile.write("\n")
    # elif fpath == "../../SJP_JC_Audio/perception_results/fluency_avgs.csv":
    #     with open('../../SJP_JC_Audio/gridsearch/phonetic_IS09-full_lstm_fluency.csv', 'w') as wfile:
    #         for item in total_stats:
    #             wfile.write(",".join(item))
    #             wfile.write("\n")
    # else:
    #     with open('../../SJP_JC_Audio/gridsearch/phonetic_IS09-full_lstm_accentedness.csv', 'w') as wfile:
    #         for item in total_stats:
    #             wfile.write(",".join(item))
    #             wfile.write("\n")