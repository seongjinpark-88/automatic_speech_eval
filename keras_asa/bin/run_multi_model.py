# run the code created in multi_input_model.py
import numpy as np
import multi_input_model as modeler
import matplotlib.pyplot as plt

######################################################
############ RUN INITIAL PHONETIC TEST ###############

# create an instance of GetFeatures class
phonetic_test = modeler.GetFeatures("../../SJP_JC_Audio/wavs", "~/opensmile-2.3.0", "../../SJP_JC_Audio/IS10_smallset")
# uncomment to extract features for first time use
# phonetic_test.extract_features()

# get dict with all features for the dataset
# acoustic_features = phonetic_test.get_features_dict()

# selected columns in IS10
acoustic_features = phonetic_test.get_features_dict(dropped_cols=['name', 'frameTime',
                                                                  "mfcc_sma[0]", "mfcc_sma[1]", "mfcc_sma[2]",
                                                                  "mfcc_sma[3]", "mfcc_sma[4]", "mfcc_sma[5]",
                                                                  "mfcc_sma[6]", "mfcc_sma[7]", "mfcc_sma[8]",
                                                                  "mfcc_sma[9]", "mfcc_sma[10]", "mfcc_sma[11]",
                                                                  "mfcc_sma[12]", "mfcc_sma[13]", "mfcc_sma[14]",
                                                                  "logMelFreqBand_sma[0]", "logMelFreqBand_sma[1]",
                                                                  "logMelFreqBand_sma[2]", "logMelFreqBand_sma[3]",
                                                                  "logMelFreqBand_sma[4]", "logMelFreqBand_sma[5]",
                                                                  "logMelFreqBand_sma[6]", "logMelFreqBand_sma[7]",
                                                                  "lspFreq_sma[0]", "lspFreq_sma[1]", "lspFreq_sma[2]",
                                                                  "lspFreq_sma[3]", "lspFreq_sma[4]", "lspFreq_sma[5]",
                                                                  "lspFreq_sma[6]", "lspFreq_sma[7]", "mfcc_sma_de[0]",
                                                                  "mfcc_sma_de[1]", "mfcc_sma_de[2]", "mfcc_sma_de[3]",
                                                                  "mfcc_sma_de[4]", "mfcc_sma_de[5]", "mfcc_sma_de[6]",
                                                                  "mfcc_sma_de[7]", "mfcc_sma_de[8]", "mfcc_sma_de[9]",
                                                                  "mfcc_sma_de[10]", "mfcc_sma_de[11]", "mfcc_sma_de[12]",
                                                                  "mfcc_sma_de[13]", "mfcc_sma_de[14]",
                                                                  "logMelFreqBand_sma_de[0]", "logMelFreqBand_sma_de[1]",
                                                                  "logMelFreqBand_sma_de[2]", "logMelFreqBand_sma_de[3]",
                                                                  "logMelFreqBand_sma_de[4]", "logMelFreqBand_sma_de[5]",
                                                                  "logMelFreqBand_sma_de[6]", "logMelFreqBand_sma_de[7]",
                                                                  "lspFreq_sma_de[0]", "lspFreq_sma_de[1]",
                                                                  "lspFreq_sma_de[2]", "lspFreq_sma_de[3]",
                                                                  "lspFreq_sma_de[4]", "lspFreq_sma_de[5]",
                                                                  "lspFreq_sma_de[6]", "lspFreq_sma_de[7]"])

# selected columns in IS09
# acoustic_features = phonetic_test.get_features_dict(dropped_cols=['name', 'frameTime',
#                                                                   "pcm_RMSenergy_sma_de","pcm_fftMag_mfcc_sma_de[1]",
#                                                                   "pcm_fftMag_mfcc_sma_de[2]","pcm_fftMag_mfcc_sma_de[3]",
#                                                                   "pcm_fftMag_mfcc_sma_de[4]",
#                                                                   "pcm_fftMag_mfcc_sma_de[5]","pcm_fftMag_mfcc_sma_de[6]",
#                                                                   "pcm_fftMag_mfcc_sma_de[7]","pcm_fftMag_mfcc_sma_de[8]",
#                                                                   "pcm_fftMag_mfcc_sma_de[9]","pcm_fftMag_mfcc_sma_de[10]",
#                                                                   "pcm_fftMag_mfcc_sma_de[11]","pcm_fftMag_mfcc_sma_de[12]",
#                                                                   "pcm_zcr_sma_de","voiceProb_sma_de","F0_sma_de"])

# select participants to use
speaker_list = ["S02", "S03", "S04", "S05", "S07", "S08", "S09", "S19", "S21", "S22", "S23", "S24", "S25", "S26", "S28"]
# speaker_list = ["S07"]  # fixme: for testing--this doesn't actually work as expected, so something is hard-coded

# read in y data
y_values = modeler.get_ys_dict("../../SJP_JC_Audio/perception_results/accented_avgs.csv",
                                     speaker_list)

# zip x and y data
zipped = modeler.zip_feats_and_ys(acoustic_features, y_values)

# test to make sure this worked
unzipped_feats, unzipped_ys = zip(*zipped)

# set variables for input into model
unzipped_ys = list(unzipped_ys)
unzipped_feats = np.array(list(unzipped_feats))
shape = unzipped_feats.shape

# create instance of class AdaptiveModel
adapt = modeler.AdaptiveModel(unzipped_feats, unzipped_ys, shape, "../../SJP_JC_Audio/IS10_smallset")

# split data into datasets
trainX, trainy, valX, valy, testX, testy = adapt.split_data()

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

                valy, y_preds = modeler.train_and_predict(model, trainX, trainy, valX, valy, batch=b)

                plt.plot([0, 6], [0, 6], 'k-', color='red')
                plt.scatter(valy, y_preds, color='blue', label='data')

                plt.legend()
                plt.xlabel('observed (y)')
                plt.xlim(0, 6)
                plt.ylabel('predicted (y-hat)')
                plt.ylim(0, 6)

                plt.title('y vs y-hat {0} lstm units, {1} connected units, {2} lstm layers, batch size {3}'.format(n_l_u, n_c_u, n_l, b))
                plt.savefig("smallset_dev-loss_{0}-{1}-{2}-{3}.png".format(n_l_u, n_c_u, n_l, b))
                plt.clf()


######################################################
############ RUN SUMMARY PHONETIC TEST ###############
# # one set of input per data point -- NOT RNN; use MLP instead
#
# # create an instance of GetFeatures class
# summary_test = modeler.GetFeatures("../../SJP_JC_Audio/wavs", "~/opensmile-2.3.0", "../../SJP_JC_Audio/IS10_summary_test")
# # uncomment to extract features for first time use
# summary_test.extract_features(summary_stats=True)
#
# # get dict with all features for the dataset
# summary_features = summary_test.get_features_dict()
#
# # select participants to use
# speaker_list = ["S02", "S03", "S04", "S05", "S07", "S08", "S09", "S19", "S21", "S22", "S23", "S24", "S25", "S26", "S28"]
# # speaker_list = ["S07"]  # fixme: for testing--this doesn't actually work as expected, so something is hard-coded
#
# # read in y data
# summary_y_values = modeler.get_ys_dict("../../SJP_JC_Audio/perception_results/accented_avgs.csv",
#                                     speaker_list)
#
# # zip x and y data
# summary_zipped = modeler.zip_feats_and_ys(summary_features, summary_y_values)
#
# # test to make sure this worked
# unzipped_summary_feats, unzipped_summary_ys = zip(*summary_zipped)
#
# # set variables for input into model
# unzipped_summary_ys = list(unzipped_summary_ys)
# unzipped_summary_feats = np.array(list(unzipped_summary_feats))
# summary_shape = unzipped_summary_feats.shape
#
# # create instance of class AdaptiveModel
# sum_adapt = modeler.AdaptiveModel(unzipped_summary_feats, unzipped_summary_ys, summary_shape, "../../SJP_JC_Audio/IS10_summary_test")
#
# # split data into datasets
# train_summ_X, train_summ_y, val_summ_X, val_summ_y, _, _ = sum_adapt.split_data()
#
# train_summ_X = modeler.reshape_data(train_summ_X)
# val_summ_X = modeler.reshape_data(val_summ_X)
#
# # hyperparameter tuning on MLP model
# n_connected_units = [5, 10, 20, 50, 100, 200, 400] # higher than 5 is better...
# n_connected = [2, 3, 4, 5, 10] # larger numbers fare worse
# batch = [12, 32, 64, 100]
#
# # # Recursive Feature Elimination
# # from sklearn import datasets
# # from sklearn.feature_selection import RFE
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.feature_selection import SelectKBest
# # from sklearn.feature_selection import f_regression
# #
# # # create a base classifier used to evaluate a subset of attributes
# # model = LogisticRegression()
# # # create the RFE model and select 3 attributes
# # # rfe = RFE(model, 50)
# # tester = SelectKBest(score_func=f_regression(train_summ_X, train_summ_y), k=50)
# # fit = tester.fit(train_summ_X, train_summ_y)
# # print(fit.scores_)
# # print(fit.pvalues_)
# # features = fit.transform(train_summ_X)
# # # summarize selected features
# # print(features[0:51,:])
# # # rfe = rfe.fit(train_summ_X, train_summ_y)
# # # summarize the selection of the attributes
# # print(rfe.support_)
# # print(rfe.ranking_)
# # sys.exit(1)
#
#
# # moderate-sized grid search
# for n_c_u in n_connected_units:
#     for n_c in n_connected:
#         for b in batch:
#
#             print("Number of connected units: " + str(n_c_u))
#             print("Number of connected layers: " + str(n_c))
#             print("Batch size: " + str(b))
#
#             sum_model = sum_adapt.mlp_model(n_connected=n_c, n_connected_units=n_c_u, act='tanh', output_act='linear',
#                                             output_size=1)
#
#             valy, y_preds = modeler.train_and_predict(sum_model, train_summ_X, train_summ_y, val_summ_X,
#                                                       val_summ_y, batch=b)
#
#             plt.plot([0, 6], [0, 6], 'k-', color='red')
#             plt.scatter(valy, y_preds, color='blue', label='data')
#
#             plt.legend()
#             plt.xlabel('observed (y)')
#             plt.xlim(0, 6)
#             plt.ylabel('predicted (y-hat)')
#             plt.ylim(0, 6)
#
#             plt.title('y vs y-hat {0} connected units, {1} connected layers, batch size {2}'.format(n_c_u, n_c, b))
#             plt.savefig("IS10_mlp_dev-loss_{0}-{1}-{2}.png".format(n_c_u, n_c, b))
#             plt.clf()
