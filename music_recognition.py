import os
from keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
import numpy as np

import librosa

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal',
          'pop', 'reggae', 'rock']
WINDOW_SIZE = 2048
WINDOW_STRIDE = WINDOW_SIZE // 2
N_MELS = 128
MEL_KWARGS = {
    'n_fft': WINDOW_SIZE,
    'hop_length': WINDOW_STRIDE,
    'n_mels': N_MELS
}


class Music_Recognition(object):

    def __init__(self):
        N_LAYERS = 3
        FILTER_LENGTH = 5
        CONV_FILTER_COUNT = 256
        n_features = 128
        input_shape = (None, n_features)
        model_input = Input(input_shape, name='input')
        layer = model_input
        for i in range(N_LAYERS):
            layer = Convolution1D(
                filters=CONV_FILTER_COUNT,
                kernel_size=FILTER_LENGTH,
                name='convolution_' + str(i + 1)
            )(layer)
            layer = BatchNormalization(momentum=0.9)(layer)
            layer = Activation('relu')(layer)
            layer = MaxPooling1D(2)(layer)
            layer = Dropout(0.5)(layer)

        layer = TimeDistributed(Dense(len(GENRES)))(layer)
        time_distributed_merge_layer = Lambda(
            function=lambda x: K.mean(x, axis=1),
            output_shape=lambda shape: (shape[0],) + shape[2:],
            name='output_merged'
        )
        layer = time_distributed_merge_layer(layer)
        layer = Activation('softmax', name='output_realtime')(layer)
        model_output = layer
        self.model = Model(model_input, model_output)
        self.model.load_weights(os.path.join('models', 'model.h5'))
        print('load weight')
        self.analysis(os.path.join('upload', 'test.wav'))
        print('test success')

    def analysis(self, file_path):
        y, sr = librosa.load(file_path, mono=True)

        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        times = librosa.frames_to_time(beats, sr=sr)
        print('get beats')

        melspec = librosa.feature.melspectrogram(y, sr)
        logmelspec = librosa.power_to_db(melspec)
        db_list = [logmelspec[0:120:3, beat].tolist() for beat in beats]
        print('get db list')

        label = self.get_style(y)
        print('get label')

        return times.tolist(), db_list, label

    def cut_music(self, time, file_name='lhls.wav', out_file_name='test.wav'):
        from pydub import AudioSegment
        audio = AudioSegment.from_wav(file_name)
        audio[:time * 1000].export(out_file_name, format="wav")

    def get_style(self, y):
        features = librosa.feature.melspectrogram(y, **MEL_KWARGS).T
        features[features == 0] = 1e-6
        features = features[np.newaxis, :]
        features = np.log(features)

        label = self.model.predict(features)[0]
        max_index = int(np.argmax(label))
        GENRES_label = GENRES[max_index]
        return GENRES_label


if __name__ == '__main__':
    m = Music_Recognition()
    m.analysis('upload/test.wav')
