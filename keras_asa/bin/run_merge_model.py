from utils import models_update
import numpy as np

from sklearn.model_selection import train_test_split

from numba import cuda

from matplotlib import pyplot as plt

wav2idx, wav_names, melspec_dict, mfcc_dict = models_update.get_audio_features("../audio/wavs/")

phonetic_test = models_update.GetFeatures("../audio/wavs", "~/opensmile-2.3.0", "../audio/IS09_summary")

# phonetic_test.extract_features(summary_stats=True)

acoustic_features = phonetic_test.get_features_dict()

# acoustic_features = phonetic_test.get_features_dict(dropped_cols=['name', 'frameTime',
#                                                                   "mfcc_sma[0]", "mfcc_sma[1]", "mfcc_sma[2]",
#                                                                   "mfcc_sma[3]", "mfcc_sma[4]", "mfcc_sma[5]",
#                                                                   "mfcc_sma[6]", "mfcc_sma[7]", "mfcc_sma[8]",
#                                                                   "mfcc_sma[9]", "mfcc_sma[10]", "mfcc_sma[11]",
#                                                                   "mfcc_sma[12]", "mfcc_sma[13]", "mfcc_sma[14]",
#                                                                   "logMelFreqBand_sma[0]", "logMelFreqBand_sma[1]",
#                                                                   "logMelFreqBand_sma[2]", "logMelFreqBand_sma[3]",
#                                                                   "logMelFreqBand_sma[4]", "logMelFreqBand_sma[5]",
#                                                                   "logMelFreqBand_sma[6]", "logMelFreqBand_sma[7]",
#                                                                   "lspFreq_sma[0]", "lspFreq_sma[1]", "lspFreq_sma[2]",
#                                                                   "lspFreq_sma[3]", "lspFreq_sma[4]", "lspFreq_sma[5]",
#                                                                   "lspFreq_sma[6]", "lspFreq_sma[7]", "mfcc_sma_de[0]",
#                                                                   "mfcc_sma_de[1]", "mfcc_sma_de[2]", "mfcc_sma_de[3]",
#                                                                   "mfcc_sma_de[4]", "mfcc_sma_de[5]", "mfcc_sma_de[6]",
#                                                                   "mfcc_sma_de[7]", "mfcc_sma_de[8]", "mfcc_sma_de[9]",
#                                                                   "mfcc_sma_de[10]", "mfcc_sma_de[11]", "mfcc_sma_de[12]",
#                                                                   "mfcc_sma_de[13]", "mfcc_sma_de[14]",
#                                                                   "logMelFreqBand_sma_de[0]", "logMelFreqBand_sma_de[1]",
#                                                                   "logMelFreqBand_sma_de[2]", "logMelFreqBand_sma_de[3]",
#                                                                   "logMelFreqBand_sma_de[4]", "logMelFreqBand_sma_de[5]",
#                                                                   "logMelFreqBand_sma_de[6]", "logMelFreqBand_sma_de[7]",
#                                                                   "lspFreq_sma_de[0]", "lspFreq_sma_de[1]",
#                                                                   "lspFreq_sma_de[2]", "lspFreq_sma_de[3]",
#                                                                   "lspFreq_sma_de[4]", "lspFreq_sma_de[5]",
#                                                                   "lspFreq_sma_de[6]", "lspFreq_sma_de[7]"])

# Get mfccs and melspec features
X, Y, X_wav = models_update.get_data("../data/perception_results/fluency_avgs.csv", 
    wav2idx, melspec_dict, acoustic = False)

# Get acoustic features
X_phon, Y, X_wav = models_update.get_data("../data/perception_results/fluency_avgs.csv", 
    wav2idx, acoustic_features, acoustic = True)

# Reshape data for lstm
X = np.reshape(X, (np.shape(X)[0], np.shape(X)[2], np.shape(X)[1]))

# Reshape data for mlp
X_phon = np.reshape(X_phon, (np.shape(X_phon)[0], np.shape(X_phon)[-1]))

# # Pad and normalize opensmile features (time-stamped)
# X_phon = models_update.pad_feats(X_phon, normalize = True)

print("X shape: ", np.shape(X))
print("X phon. shape: ", np.shape(X_phon))

# exit()

# X_train, X_test, X_phon_train, X_phon_test, X_wav_train, X_wav_test, Y_train, Y_test = train_test_split(
#   X, X_phon, X_wav, Y, test_size = 0.3, random_state = 77)


# CV indices
CV_IDX = models_update.get_cv_index(10, X)

# lists to save the results
CV_mse = []
CV_histories = []
CV_prediction = []


cv_idx = 1

for train_index, test_index in CV_IDX:
    # print("TRAIN: ", X_wav[train_index][:1])
    # print("TRAIN: ", Y[test_index][:10])

    # Create test and training data
    print("Cross-validation idx: ", cv_idx)

    X_train, X_test = X[train_index], X[test_index]
    X_train_phon, X_test_phon = X_phon[train_index], X_phon[test_index]
    X_train_wav, X_test_wav = X_wav[train_index], X_wav[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    # inner_cv = models_update.get_cv_index(10, X_train)
    # inner_idx = 1

    # for tr_idx, te_idx in inner_cv:
    #     print("Cross-validation idx: %d-%d" % (cv_idx, inner_idx))
    #     X_inner_tr, X_inner_val = X_train[tr_idx], X_train[te_idx]
    #     X_inner_ph_tr, X_inner_ph_val = X_train_phon[tr_index], X_train_phon[te_index]
    #     X_inner_wav_tr, X_inner_wav_val = X_train_wav[tr_index], X_train_wav[te_index]
    #     y_inner_tr, y_inner_te = y_train[tr_index], y_train[te_index]

    # Create acoustic model
    phon_model = models_update.Models(X_train_phon, "phonetic_model", model_type = "mlp")
    # phon_lstm = phon_model.bi_lstm_model(n_lstm_units = 512, n_connected_units = 512)
    phon_mlp = phon_model.mlp_model(n_connected_units = 512)

    # Create raw signal model
    raw_model = models_update.Models(X_train, "raw_acoustic", model_type = "lstm")
    raw_lstm = raw_model.bi_lstm_model(n_lstm_units = 512, n_connected_units = 512)

    # Merge two models
    merged_model = models_update.MergeModels(input_models = [phon_mlp, raw_lstm], 
        input_layers = [phon_model.mlp_input, raw_model.lstm_input])

    # Add final FC layer and compile the model
    merged_model.final_layers(n_connected_units = 128)
    merged_model.compile_model(l_rate = 0.0005)

    # Define dataset
    input_features = {'phonetic_model': X_train_phon, 'raw_acoustic': X_train}
    test_features = {'phonetic_model': X_test_phon, 'raw_acoustic': X_test}
    validation_set = ({'phonetic_model': X_test_phon, 'raw_acoustic': X_test}, {'final_output': y_test})

    # Model name to save
    model_name = "raw-phon_merge_CV%d_flu" % cv_idx

    # Train the model
    history = merged_model.train_model(epochs = 100, batch_size = 32, 
        input_feature = input_features, output_label = y_train, 
        validation = validation_set, model_name = model_name)

    # Append history
    CV_histories.append(history)

    # Save CV model
    merged_model.save_model("../data/models/raw_phon_merge_CV%d_flu.h5" % cv_idx)

    # Make a prediction
    y_prediction = merged_model.predict_model(input_feature = test_features)

    # Get MSE and r-squared
    scores = merged_model.evaluate_model(test_features, y_test)

    # Save MSE
    CV_mse.append(scores)

    # Save final prediction results
    for i in range(len(y_test)):
        result = "%d\t%s\t%d\t%f\n" % (cv_idx, wav_names[X_test_wav[i]], y_test[i], y_prediction[i][0])
#         print(result)
        CV_prediction.append(result)
    
    cv_idx += 1

    # Final evaluation of the model
    print("SCORE: ", scores)

# Save 10CV results
with open("../results/raw_phon_10CV_flu_wVal.txt", "w") as output:
    header = "CV\tstimulus\ttrue\tpred\n"
    output.write(header)
    for prediction in CV_prediction:
        output.write(prediction)

# Print MSE and r-squared for each CV
for i, mse in enumerate(CV_mse[-10:]):
    print(i + 1, mse)