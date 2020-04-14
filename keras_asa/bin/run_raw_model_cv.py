from utils import models_update
from utils import lstmCrfMtl

import numpy as np

wav2idx, wav_names, melspec_dict, mfcc_dict = models_update.get_audio_features("../audio/wavs/")

X, Y_acc, X_wav = models_update.get_data("../data/perception_results/accented_avgs.csv", wav2idx, melspec_dict)
_, Y_flu, _ = models_update.get_data("../data/perception_results/fluency_avgs.csv", wav2idx, melspec_dict)
_, Y_comp, _ = models_update.get_data("../data/perception_results/comp_avgs.csv", wav2idx, melspec_dict)

CV_IDX = models_update.get_cv_index(10, X)

CV_mse = []
CV_histories = []
CV_prediction = []

cv_idx = 1



for train_index, test_index in CV_IDX:
    # print("TRAIN: ", X_wav[train_index][:1])
    # print("TRAIN: ", Y[test_index][:10])

    print("Cross-validation idx: ", cv_idx)
    X_train, X_test = X[train_index], X[test_index]
    X_train_wav, X_test_wav = X_wav[train_index], X_wav[test_index]
    y_acc_train, y_acc_test = Y_acc[train_index], Y_acc[test_index]
    y_flu_train, y_flu_test = Y_flu[train_index], Y_flu[test_index]
    y_comp_train, y_comp_test = Y_comp[train_index], Y_comp[test_index]

    # model = models.Models(X)

    # lstm_model = model.bi_lstm_model(n_connected = 1, n_connected_units = 128)
    #
    # history = lstm_model.fit(X_train, y_train, batch_size = 32, epochs = 80, shuffle = False,
    #                    class_weight = None, verbose = 1, validation_data = (X_test, y_test))
    #

    model = lstmCrfMtl.lstmMtl(X_train, name="mtl", model_type="lstm")
    model.bi_lstm_model(n_lstm_units=512, dropout=0.2, n_connected_units=128, act='tanh')
    # lstmMtl.compile_model(l_rate=0.0001)

    history = model.train_model(epochs = 100, batch_size = 64,
        input_feature = X_train, output_label = [y_acc_train, y_flu_train, y_comp_train], model_name = "lstmMtl")

    CV_histories.append(history)
    
    scores = model.evaluate_model(X_test, [y_acc_test, y_flu_test, y_comp_test])
    CV_mse.append(scores[1])
    # prediction_type: 0 (acc), 1 (flu), 2 (comp)
    y_prediction = model.predict_model(input_feature=X_test, prediction_type=0)
    
    for i in range(len(y_acc_test)):
        result = "%d\t%s\t%s\t%s\n" % (cv_idx, wav_names[X_test_wav[i]], y_acc_test[i], y_prediction[i])
        print(result)
        CV_prediction.append(result)
    
    cv_idx += 1

    # Final evaluation of the model
    # scores = lstm_model.evaluate(X_test, y_test, verbose=0)
    print("MSE: ", scores)

with open("../results/mel_10CV_fl_dropout_Mtl.txt", "w") as output:
    for prediction in CV_prediction:
        output.write(prediction)

for i, mse in enumerate(CV_mse[-10:]):
    print(i + 1, mse)