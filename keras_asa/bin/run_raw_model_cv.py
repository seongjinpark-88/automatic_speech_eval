from utils import models
import numpy as np

from numba import cuda

cuda.select_device(0)
cuda.close()


wav2idx, wav_names, melspec_dict, mfcc_dict = models.get_audio_features("../audio/wavs/")

X, Y, X_wav = models.get_data("../data/perception_results/comp_data.csv", wav2idx, melspec_dict)

CV_IDX = models.get_cv_index(10, X)

CV_mse = []
CV_histories = []
CV_prediction = []

cv_idx = 1

for train_index, test_index in CV_IDX:
    # print("TRAIN: ", X_wav[train_index][:1])
    # print("TRAIN: ", Y[test_index][:10])

    print(cv_idx)
    X_train, X_test = X[train_index], X[test_index]
    X_train_wav, X_test_wav = X_wav[train_index], X_wav[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    model = models.Models(X)

    lstm_model = model.bi_lstm_model(n_connected = 1, n_connected_units = 128)
    
    history = lstm_model.fit(X_train, y_train, batch_size = 32, epochs = 40, shuffle = False,
                       class_weight = None, verbose = 1, validation_data = (X_test, y_test))
                       
    CV_histories.append(history)
    
    scores = lstm_model.evaluate(X_test, y_test, verbose=0)
    CV_mse.append(scores[1])
    y_prediction = lstm_model.predict(X_test)
    
    for i in range(len(y_test)):
        result = "%d\t%s\t%d\t%f\n" % (cv_idx, wav_names[X_test_wav[i]], y_test[i], y_prediction[i][0])
#         print(result)
        CV_prediction.append(result)
    
    cv_idx += 1

    # Final evaluation of the model
    scores = lstm_model.evaluate(X_test, y_test, verbose=0)
    print("MSE: %.5f" % (scores[1]))

with open("../results/mel_10CV_comp.txt", "w") as output:
    for prediction in CV_prediction:
        output.write(prediction)

for i, mse in enumerate(CV_mse[-10:]):
    print(i + 1, mse)


# lstm_model = model.bi_lstm_model(n_connected = 1, n_connected_units = 128)

# mses, histories, predictions = model.cv_train_and_predict(batch_size = 64, 
# 	num_epochs = 40, X = X, Y = Y, X_wav = X_wav, wav_names = wav_names, 
# 	cv_index = CV_IDX, output_name = "comp_10CV_mel.txt")

