# run the code created in multi_input_model.py
import numpy as np
import multi_input_model as modeler

# create an instance of GetFeatures class
phonetic_test = modeler.GetFeatures("../../SJP_JC_Audio/wavs", "~/opensmile-2.3.0", "../../SJP_JC_Audio/output")
# uncomment to extract features for first time use
# phonetic_test.extract_features()

# get dict with all features for the dataset
acoustic_features = phonetic_test.get_features_dict(dropped_cols=['name', 'frameTime',
                                                                  "pcm_RMSenergy_sma_de","pcm_fftMag_mfcc_sma_de[1]",
                                                                  "pcm_fftMag_mfcc_sma_de[2]","pcm_fftMag_mfcc_sma_de[3]",
                                                                  "pcm_fftMag_mfcc_sma_de[4]",
                                                                  "pcm_fftMag_mfcc_sma_de[5]","pcm_fftMag_mfcc_sma_de[6]",
                                                                  "pcm_fftMag_mfcc_sma_de[7]","pcm_fftMag_mfcc_sma_de[8]",
                                                                  "pcm_fftMag_mfcc_sma_de[9]","pcm_fftMag_mfcc_sma_de[10]",
                                                                  "pcm_fftMag_mfcc_sma_de[11]","pcm_fftMag_mfcc_sma_de[12]",
                                                                  "pcm_zcr_sma_de","voiceProb_sma_de","F0_sma_de"])

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


# uncomment the following lines to save and load folds
# adapt.save_data(trainX, trainy, valX, valy, testX, testy)
# trainX, trainy, valX, valy, testX, testy = adapt.load_existing_data("X_train.npy", "y_train.npy",
#                                                                     "X_val.npy", "y_val.npy",
#                                                                     "X_test.npy", "y_test.npy")

# hyperparameter tuning on LSTM model
n_lstm_units = [5, 10] #[20, 50, 100]  # 10 done
n_connected_units = [5, 10]  # , 20, 50]
# n_lstm = [1, 5, 10] # 20 seemed really bad
n_lstm = [1, 2]
batch = [32, 64]

# prep pyplot to view the results
import matplotlib.pyplot as plt

adapt = modeler.AdaptiveModel(unzipped_feats, unzipped_ys, shape, "../../SJP_JC_Audio/output")

# split data into datasets
trainX, trainy, valX, valy, testX, testy = adapt.split_data()

# moderate-sized grid search
for n_l_u in n_lstm_units:
    for n_c_u in n_connected_units:
        for n_l in n_lstm:
            for b in batch:
                # todo: optimize this by removing model from adapt (use as input to training function,
                #       instantiate with each of the separate model calls)

                print("Number of lstm units: " + str(n_l_u))
                print("Number of connected units: " + str(n_c_u))
                print("Number of lstm layers: " + str(n_l))
                print("Batch size: " + str(b))

                model = adapt.lstm_model(n_lstm=n_l, output_size=1, n_lstm_units=n_l_u, n_connected_units=n_c_u,
                                 act="tanh")  # relu doesn't work with this dataset

                valy, y_preds = adapt.train_and_predict(model, trainX, trainy, valX, valy, batch=b)

                plt.plot([0, 6], [0, 6], 'k-', color='red')
                plt.scatter(valy, y_preds, color='blue', label='data')

                plt.legend()
                plt.xlabel('observed (y)')
                plt.xlim(0, 6)
                plt.ylabel('predicted (y-hat)')
                plt.ylim(0, 6)

                plt.title('y vs y-hat {0} lstm units, {1} connected units, {2} lstm layers, batch size {3}'.format(n_l_u, n_c_u, n_l, b))
                plt.savefig("train-loss_de-dropped_{0}-{1}-{2}-{3}.png".format(n_l_u, n_c_u, n_l, b))
                plt.clf()
