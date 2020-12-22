import tensorflow as tf
import argparse
import numpy as np
from tensorflow import keras
import pandas as pd
import os
from scipy import signal
import tensorflow_model_optimization as tfmot
import tensorflow.lite as tflite
from tensorflow_model_optimization.python.core.api.sparsity import keras as sparsity
import zlib
import sys
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True, help='Version of the model')

args, _ = parser.parse_known_args()

version = args.version

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

sampling_rate = 16000

#TODO: check the size and put it as options
if version == "a" or version == "b":
    alpha = 0.3
    frame_length = 640
    frame_step = 320
    resampling_rate = None
    tensor_spec_dimension = [1, 49, 10, 1]


elif version == "c":
    alpha = 0.5
    frame_length = 240
    frame_step = 120
    resampling_rate = 8000
    tensor_spec_dimension = [1, 65, 10, 1]


else:
    print(f"Select between version a, b or c. Version {version} does not exist")
    sys.exit()


ROOT_DIR = "./"
dataset_dir= ROOT_DIR + 'kws_test_{}'.format(version)
saved_model_dir = './models/kws'
tflite_model_dir = './Group1_kws_{}.tflite.zlib'.format(version)

zip_path = tf.keras.utils.get_file(
        origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        fname='mini_speech_commands.zip',
        extract=True,
        cache_dir='.', cache_subdir='data')

data_dir = os.path.join('.', 'data', 'mini_speech_commands')
filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)


train_files = tf.strings.split(tf.io.read_file(ROOT_DIR +'kws_train_split.txt'),sep='\n')[:-1]
val_files= tf.strings.split(tf.io.read_file(ROOT_DIR +'kws_val_split.txt'),sep='\n')[:-1]
test_files = tf.strings.split(tf.io.read_file(ROOT_DIR +'kws_test_split.txt'),sep='\n')[:-1]

LABELS = np.array(tf.io.gfile.listdir(str(data_dir)))
LABELS = LABELS[LABELS != 'README.md']


#Data preprocessing
class SignalGenerator:
    def __init__(self, labels, sampling_rate, frame_length, frame_step,
            num_mel_bins=None, lower_frequency=None, upper_frequency=None,
            num_coefficients=None, mfcc=False, resampling_rate = None):

        self.labels = labels

        self.sampling_rate = sampling_rate
        self.resampling_rate = resampling_rate

        self.frame_length = frame_length
        self.frame_step = frame_step

        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency

        self.num_mel_bins = num_mel_bins
        self.num_coefficients = num_coefficients

        num_spectrogram_bins = (frame_length) // 2 + 1


        if self.resampling_rate is not None:
            rate = self.resampling_rate

        else:
            rate = self.sampling_rate
           
        if mfcc is True:
            self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                    self.num_mel_bins, num_spectrogram_bins, rate,
                    self.lower_frequency, self.upper_frequency)
            self.preprocess = self.preprocess_with_mfcc
        else:
            self.preprocess = self.preprocess_with_stft


    def custom_resampling(self, audio):
        audio = signal.resample_poly(audio, 1, self.sampling_rate // self.resampling_rate)
        audio = tf.convert_to_tensor(audio, dtype=tf.float32)
        return audio

    def read(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2]
        label_id = tf.argmax(label == self.labels)
        audio_binary = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio_binary)
        audio = tf.squeeze(audio, axis=1)

        if self.resampling_rate is not None:
            audio = tf.numpy_function(self.custom_resampling, [audio], tf.float32)

        return audio, label_id


    def pad(self, audio):
        if self.resampling_rate is not None:
            rate = self.resampling_rate
        else:
            rate = self.sampling_rate
        zero_padding = tf.zeros([rate] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([rate])

        return audio

    def get_spectrogram(self, audio):
        stft = tf.signal.stft(audio, frame_length=self.frame_length,
                frame_step=self.frame_step, fft_length=self.frame_length)
        spectrogram = tf.abs(stft)

        return spectrogram

    def get_mfccs(self, spectrogram):
        mel_spectrogram = tf.tensordot(spectrogram,
                self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.num_coefficients]

        return mfccs

    def preprocess_with_stft(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        spectrogram = tf.image.resize(spectrogram, [32, 32])

        return spectrogram, label

    def preprocess_with_mfcc(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        mfccs = self.get_mfccs(spectrogram)
        mfccs = tf.expand_dims(mfccs, -1)

        return mfccs, label

    def make_dataset(self, files, train):
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.map(self.preprocess, num_parallel_calls=4)
        ds = ds.batch(32)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds



MFCC_OPTIONS = {'frame_length': frame_length, 'frame_step': frame_step, 'mfcc': True,
        'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40,
        'num_coefficients': 10}


options = MFCC_OPTIONS
strides = [2, 1]
units = 8

generator = SignalGenerator(LABELS, sampling_rate=sampling_rate, resampling_rate=resampling_rate, **options)
train_ds = generator.make_dataset(train_files, True)
val_ds = generator.make_dataset(val_files, False)
test_ds = generator.make_dataset(test_files, False)


if os.path.exists(dataset_dir):
    shutil.rmtree(dataset_dir)

tf.data.experimental.save(test_ds, dataset_dir)


# Model selection and trainig
model = keras.Sequential([
    keras.layers.Conv2D(filters=int(alpha*256), kernel_size=[3, 3], strides=strides, use_bias=False),
    keras.layers.BatchNormalization(momentum=0.1),
    keras.layers.ReLU(),
    keras.layers.DepthwiseConv2D(kernel_size=[3,3],strides=[1, 1], use_bias=False),
    keras.layers.Conv2D(filters=int(alpha*256), kernel_size=[1, 1], strides=[1,1], use_bias=False),
    keras.layers.BatchNormalization(momentum=0.1),
    keras.layers.ReLU(),
    keras.layers.DepthwiseConv2D(kernel_size=[3,3], strides=[1, 1], use_bias=False),
    keras.layers.Conv2D(filters=int(alpha*256), kernel_size=[1, 1], strides=[1, 1], use_bias=False),
    keras.layers.BatchNormalization(momentum=0.1),
    keras.layers.ReLU(),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(units=8)
    ])


model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

history = model.fit(
    train_ds,
    epochs=20,
    batch_size=32,
    validation_data=val_ds,
    )

print("Test accuracy:")
test_accuracy= model.evaluate(test_ds)

model.save(saved_model_dir)


# Post training quantization and tflite conversion
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()


# Compression and saving
with open(tflite_model_dir, 'wb') as fp:
    tflite_compressed = zlib.compress(tflite_model)
    fp.write(tflite_compressed)

print(f"Size of compressed tflite model: {os.path.getsize(tflite_model_dir)/1024} kB")
