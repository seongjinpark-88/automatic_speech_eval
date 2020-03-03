#!/usr/bin/env python
# coding: utf-8

# In[73]:


import os
import pickle

from os.path import isdir, join
from pathlib import Path

import librosa
import librosa.display

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split


# In[22]:


audio_path = '../extract/wavs/'
print(os.listdir(audio_path))


# In[31]:


wav2idx = {}
melspec_dict = {}
wav_names = [wav for wav in os.listdir(audio_path) if wav.endswith("wav")]

max_len = 0

for i, w in enumerate(wav_names):
    wav2idx[w] = i
    wav_path = audio_path + w
    
    y, sr = librosa.load(wav_path) 
    mel_data = librosa.feature.melspectrogram(y = y, sr= sr)
    
    if (np.shape(mel_data)[1] > max_len):
        max_len = np.shape(mel_data)[1] 
    
    melspec_dict[i] = mel_data


# In[32]:


print(max_len)


# In[33]:


with open("../../result/accented_data.csv", "r") as f:
    data = f.readlines()


# In[56]:


X = []
Y = []

zeros = np.zeros(128)
for i in range(1, len(data)):
    line = data[i].rstrip()
    
    accented, stim, wav_name = line.split(",")
    wav_idx = wav2idx[wav_name]
    
    x = np.zeros(shape = (128, max_len))
    x_data = melspec_dict[wav_idx]

#     print(np.shape(x), np.shape(x_data))
    
    for i in range(0, np.shape(x_data)[0]):
        for j in range(0, np.shape(x_data)[1]):
            x[i][j] = x_data[i][j]
    
    X.append(x)
    Y.append(int(accented))

print(np.shape(X))
print(np.shape(Y))


# In[58]:


Y_onehot = to_categorical(Y)
print(np.shape(X))
print(np.shape(Y_onehot))


# In[77]:


X_train, X_test, y_train, y_test = train_test_split(X, Y_onehot, test_size=0.3, random_state=42)


# In[69]:


# hyperparameters
lr = 0.001
batch_size = 64
drop_out_rate = 0.25
num_dense_unit = 256
num_epochs = 10

num_classes = np.shape(Y_onehot)[1]
num_mel = 128
max_time = max_len

input_shape = (num_mel, max_time, 1)


# In[82]:


# reshape

X_train = np.array(X_train)
X_test = np.array(X_test)

X_train = X_train.reshape(X_train.shape[0], num_mel, max_time, 1)
X_test = X_test.reshape(X_test.shape[0], num_mel, max_time, 1)
print(np.shape(X_train))
print(np.shape(X_test))


# In[83]:


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print('X_train shape: ', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print(np.shape(y_train), np.shape(y_test))


# In[84]:


model = Sequential()
model.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu', input_shape = input_shape))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(256, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(drop_out_rate))
model.add(Flatten())

model.add(Dense(num_dense_unit, activation = 'relu'))
model.add(Dropout(drop_out_rate))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(drop_out_rate))

model.add(Dense(num_classes, activation = 'softmax'))

# use adam optimizer
adam = keras.optimizers.Adam(lr = lr)

# compile the model
model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()


# In[ ]:


model.fit(X_train, y_train, batch_size = batch_size, epochs = num_epochs, shuffle = False, class_weight = None,
         verbose = 1, validation_data = (X_test, y_test))

model.save('CNN_model.h5')


# In[50]:


S = librosa.feature.melspectrogram(y=y, sr=sr)
np.shape(S)


# In[13]:


plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(S,                                              
ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()

