import librosa
import numpy as np

import librosa

src_loaded, sr = librosa.load('filename.mp3', sr=12000, duration=29., mono=True) # 
print(src.shape) # (N, )

if src.shape[0] < 44100 * 14:
    print('Concat zeros so that the shape becomes (617400, )')
if src.shape[0] > 44100 * 14:
    print('If you set the duration as 14 in the loading function, the shape is probably (617401, ).')
    print('In this case, trim it to make it (617400, ).')
    src = src[:617400]

# It's pretty done, now the src.shape == (348000, )
# However, Kapre, the audio preprocessing library expect something like (n_channel, length) 
#  because I wanted the `ndim` of the signal to be in a consistent format.
# So,...
src = src[np.newaxis, :] # now it's (1, 348000)

# That's it!
# But beware! You need another dimension, the batch dimension, in the model training. 
# So your final shape of the batch would be (batch_size, 1, 348000) which is not covered in this code.
# © 2020 GitHub, Inc.