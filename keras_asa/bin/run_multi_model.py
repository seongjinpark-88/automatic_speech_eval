# run the code created in multi_input_model.py
import numpy as np
import multi_input_model as modeler

# create an instance of GetFeatures class
phonetic_test = modeler.GetFeatures("../../SJP_JC_Audio/wavs", "~/opensmile-2.3.0", "../../SJP_JC_Audio/output")
# uncomment to extract features for first time use
# phonetic_test.extract_features()

# get dict with all features for the dataset
acoustic_features = phonetic_test.get_features_dict()

# select participants to use
speaker_list = ["S02", "S03", "S04", "S05", "S07", "S08", "S09", "S19", "S21", "S22", "S23", "S24", "S25", "S26", "S28"]
# speaker_list = ["S07"]  # fixme: for testing--this doesn't actually work as expected, so something is hard-coded

# read in y data
y_values = phonetic_test.get_ys_dict("../../SJP_JC_Audio/perception_results/accented_avgs.csv",
                                     speaker_list)

# zip x and y data
zipped = phonetic_test.zip_feats_and_ys(acoustic_features, y_values)

# test to make sure this worked
unzipped_feats, unzipped_ys = zip(*zipped)

# set variables for input into model
unzipped_ys = list(unzipped_ys)
unzipped_feats = np.array(list(unzipped_feats))
shape = unzipped_feats.shape

# create instance of class AdaptiveModel
adapt = modeler.AdaptiveModel(unzipped_feats, unzipped_ys, shape, "../../SJP_JC_Audio/output")

# split data into datasets
trainX, trainy, valX, valy, testX, testy = adapt.split_data()

# uncomment the following lines to save and load folds
# adapt.save_data(trainX, trainy, valX, valy, testX, testy)
# trainX, trainy, valX, valy, testX, testy = adapt.load_existing_data("X_train.npy", "y_train.npy",
#                                                                     "X_val.npy", "y_val.npy",
#                                                                     "X_test.npy", "y_test.npy")

# hyperparameter tuning on LSTM model
n_lstm_units = [10, 20, 50, 100]
n_connected_units = [5, 10, 20, 50]
n_lstm = [1, 5, 10, 20]
batch = [12, 24, 32, 64]

# moderate-sized grid search
for n_l_u in n_lstm_units:
    for n_c_u in n_connected_units:
        for n_l in n_lstm:
            for b in batch:
                print("Number of lstm units: " + str(n_l_u))
                print("Number of connected units: " + str(n_c_u))
                print("Number of lstm layers: " + str(n_l))
                print("Batch size: " + str(b))

                adapt.lstm_model(n_lstm=n_l, output_size=1, n_lstm_units=n_l_u, n_connected_units=n_c_u,
                                 act="tanh")  # relu doesn't work with this dataset

                valy, y_preds = adapt.train_and_predict(trainX, trainy, valX, valy, batch=b)

                # print the linear regression and display datapoints
                import matplotlib.pyplot as plt

                plt.scatter(valy, y_preds, color='blue', label= 'data')
                plt.plot([0,6],[0,6],'k-',color='red')

                # plt.plot(y_preds, y_fit, color='red', linewidth=2, label='Linear regression\n'+reg_label)
                plt.title('y vs y-hat {0} lstm units, {1} connected units, {2} lstm layers, batch size {3}'.format(n_l_u, n_c_u, n_l, b))
                plt.legend()
                plt.xlabel('observed (y)')
                plt.xlim(0,6)
                plt.ylabel('predicted (y-hat)')
                plt.ylim(0,6)
                plt.savefig("{0}-{1}-{2}-{3}.png".format(n_l_u, n_c_u, n_l, b))
