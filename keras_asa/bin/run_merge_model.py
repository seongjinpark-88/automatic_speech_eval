from utils import models_update
import numpy as np

from sklearn.model_selection import train_test_split

from numba import cuda

from matplotlib import pyplot as plt

wav2idx, wav_names, melspec_dict, mfcc_dict = models_update.get_audio_features("../audio/wavs/")

phonetic_test = models_update.GetFeatures("../audio/wavs", "~/opensmile-2.3.0", "../audio/IS10_smallset")

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

X, Y, X_wav = models_update.get_data("../data/perception_results/fluency_data.csv", 
	wav2idx, mfcc_dict, acoustic = False)

X_phon, Y, X_wav = models_update.get_data("../data/perception_results/fluency_data.csv", 
	wav2idx, acoustic_features, acoustic = True)

X = np.reshape(X, (np.shape(X)[0], np.shape(X)[2], np.shape(X)[1]))
X_phon = models_update.pad_feats(X_phon)

print("X shape: ", np.shape(X))
print("X phon. shape: ", np.shape(X_phon))
# exit()

X_train, X_test, X_phon_train, X_phon_test, X_wav_train, X_wav_test, Y_train, Y_test = train_test_split(
	X, X_phon, X_wav, Y, test_size = 0.3, random_state = 77)

phon_model = models_update.Models(X_phon_train, "phonetic_model")
phon_lstm = phon_model.bi_lstm_model(n_lstm_units = 512, n_connected_units = 128)

raw_model = models_update.Models(X_train, "raw_acoustic")
raw_lstm = raw_model.bi_lstm_model(n_lstm_units = 512, n_connected_units = 128)

merged_model = models_update.MergeModels(input_models = [phon_lstm, raw_lstm], 
	input_layers = [phon_model.lstm_input, raw_model.lstm_input])

merged_model.final_layers(n_connected_units = 64)
merged_model.compile_model(l_rate = 0.001)

input_features = {'phonetic_model': X_phon_train, 'raw_acoustic': X_train}
test_features = {'phonetic_model': X_phon_test, 'raw_acoustic': X_test}

history = merged_model.train_model(epochs = 100, batch_size = 64, 
	input_feature = input_features, output_label = Y_train)

merged_model.save_model("../data/models/merge_sample.h5")

pred = merged_model.predict_model(test_features)

scores = merged_model.evaluate_model(test_features, Y_test)

print("FINAL SCORE: ", score)