import tensorflow as tf
import numpy as np
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import signal
from zipfile import ZipFile
import tensorflow_model_optimization as tfmot
import tensorflow.lite as tflite
from tensorflow_model_optimization.python.core.api.sparsity import keras as sparsity
import zlib
import sys

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

ROOT_DIR = "./HMW2/"

model_type = "DS-CNN"
mfcc = True
alpha = 0.4
PRUNING = False

# zip_path = tf.keras.utils.get_file(
#         origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
#         fname='mini_speech_commands.zip',
#         extract=True,
#         cache_dir='.', cache_subdir='data')
data_dir = os.path.join('.', 'data', 'mini_speech_commands')
# filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
# filenames = tf.random.shuffle(filenames)
# num_samples = len(filenames)
# total = 8000

train_files = tf.strings.split(tf.io.read_file(ROOT_DIR +'kws_train_split.txt'),sep='\n')[:-1]
val_files= tf.strings.split(tf.io.read_file(ROOT_DIR +'kws_val_split.txt'),sep='\n')[:-1]
test_files = tf.strings.split(tf.io.read_file(ROOT_DIR +'kws_test_split.txt'),sep='\n')[:-1]

LABELS = np.array(tf.io.gfile.listdir(str(data_dir)))
LABELS = LABELS[LABELS != 'README.md']


class SignalGenerator:
    def __init__(self, labels, sampling_rate, frame_length, frame_step,
            num_mel_bins=None, lower_frequency=None, upper_frequency=None,
            num_coefficients=None, mfcc=False, resampling_rate = None):

        self.labels = labels

        # Added resampling_rte
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
            # Step for resampling
            self.step = int(self.sampling_rate/self.resampling_rate)
            
            if mfcc is True:
                self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                        self.num_mel_bins, num_spectrogram_bins, self.resampling_rate,
                        self.lower_frequency, self.upper_frequency)
                self.preprocess = self.preprocess_with_mfcc
            else:
                self.preprocess = self.preprocess_with_stft

        else:
            self.step = 1
            
            if mfcc is True:
                self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                        self.num_mel_bins, num_spectrogram_bins, self.sampling_rate,
                        self.lower_frequency, self.upper_frequency)
                self.preprocess = self.preprocess_with_mfcc
            else:
                self.preprocess = self.preprocess_with_stft

    def read(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2]
        label_id = tf.argmax(label == self.labels)
        audio_binary = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio_binary)
        audio = tf.squeeze(audio, axis=1)

        #print(self.step)

        audio = audio[::self.step]
        return audio, label_id


    def pad(self, audio):
        if self.resampling_rate is not None:
            rate = self.resampling_rate
        else:
            rate = self.sampling_rate
        zero_padding = tf.zeros([rate] - tf.shape(audio), dtype=tf.float32)
        #print(self.sampling_rate)
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
    


STFT_OPTIONS = {'frame_length': 256, 'frame_step': 128, 'mfcc': False}

# By default l=0.04 and s=0.02
# 640 = 16000 * 0.04
# 320 = 16000 * 0.02

# new_frame_length = resampling_rate * 0.04
# new_frame_step = resampling_rate * 0.02

#MFCC_OPTIONS = {'frame_length': 640, 'frame_step': 320, 'mfcc': True,
MFCC_OPTIONS = {'frame_length': 240, 'frame_step': 160, 'mfcc': True,
        'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40,
        #'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 20,
        'num_coefficients': 10}


if mfcc is True:
    options = MFCC_OPTIONS
    strides = [2, 1]
else:
    options = STFT_OPTIONS
    strides = [2, 2]


generator = SignalGenerator(LABELS, sampling_rate = 16000, resampling_rate=8000, **options)
#generator = SignalGenerator(LABELS, 8000, **options)
train_ds = generator.make_dataset(train_files, True)
val_ds = generator.make_dataset(val_files, False)
test_ds = generator.make_dataset(test_files, False)
units=8

# print(train_ds)
# sys.exit()


# for elem in val_ds:
#     print()
#     print()
#     print(elem[0].shape, elem[1].shape)
#     print()
#     print()
#     sys.exit()

# RE DO THE TEST DATASET IF WHEN CHANGING STFT OR MFCC
dataset_dir= ROOT_DIR + "/test_ds_{}".format(mfcc)
tf.data.experimental.save(test_ds, dataset_dir)

if model_type == "MLP":
    model = keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(units=int(256*alpha), activation='relu'),
        keras.layers.Dense(units=int(256*alpha), activation='relu'),
        keras.layers.Dense(units=int(256*alpha), activation='relu'),
        keras.layers.Dense(units=8)
    ])

elif model_type == "CNN-2D":
    model = keras.Sequential([
        keras.layers.Conv2D(filters=int(128*alpha), kernel_size=[3, 3], strides=strides, use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),
        keras.layers.Conv2D(filters=int(128*alpha), kernel_size=[3, 3], strides=strides, use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),
        keras.layers.Conv2D(filters=int(128*alpha), kernel_size=[3, 3], strides=strides, use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(units=8)
    ])

elif model_type == "DS-CNN":
    model = keras.Sequential([
        keras.layers.Conv2D(filters=int(alpha*256), kernel_size=[3, 3], strides=strides, use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),
        keras.layers.DepthwiseConv2D(kernel_size=[3,3],strides=[1, 1], use_bias=False),
        keras.layers.Conv2D(filters=int(alpha*256), kernel_size=[1, 1], strides=[1,1], use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),
        # Adding a dropout
        keras.layers.Dropout(0.1),
        keras.layers.DepthwiseConv2D(kernel_size=[3,3], strides=[1, 1], use_bias=False),
        keras.layers.Conv2D(filters=int(alpha*256), kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(units=8)
        ])

else:
    print("Invalid model selected")
    sys.exit()


saved_model_dir = './models/kws'

callbacks = []

if PRUNING is True:

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    pruning_params = {
        'pruning_schedule':tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.50,
            final_sparsity=0.90,
            begin_step=len(train_ds)*5,
            end_step=len(train_ds)*15),
        # 'block_size':(1, 1),
        # 'block_pooling_type':'AVG'
    }
    
    callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())

    if mfcc is True:
        input_shape =[None,49,10,1]
    
    else:
        input_shape = [None,32,32,1]

    model.build(input_shape)


model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

history = model.fit(
    train_ds,
    epochs=20,
    batch_size=32,
    validation_data=val_ds,
    callbacks = callbacks
    )

print("Test accuracy:")
test_accuracy= model.evaluate(test_ds)

model.summary()

if PRUNING is True:
    model=tfmot.sparsity.keras.strip_pruning(model)


run_model = tf.function(lambda x: model(x))

if mfcc == True:
    tensor_spec_dimension = [1, 49, 10, 1]
else:
    tensor_spec_dimension = [1, 32, 32, 1]

concrete_func = run_model.get_concrete_function(tf.TensorSpec(tensor_spec_dimension, tf.float32))
model.save(saved_model_dir, signatures=concrete_func)


# def representative_data_gen():
#   for x, _ in train_ds.take(100):
#     yield [x]

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

#converter.representative_dataset = representative_data_gen

#converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

tflite_model_dir = './models/model.tflite.zlib'

with open(tflite_model_dir, 'wb') as fp:
    tflite_compressed = zlib.compress(tflite_model)
    fp.write(tflite_compressed)

print(f"Size of compressed tflite model: {os.path.getsize(tflite_model_dir)/1024} kB")

tfModel = tflite_model_dir

str_object1 = open(tflite_model_dir, 'rb').read()
str_object2 = zlib.decompress(str_object1)
tfModel = tfModel.replace('.zlib', '')
f = open(tfModel, 'wb')
f.write(str_object2)
f.close()

print(f"Size of tflite model: {os.path.getsize(tfModel)/1024} kB")

interpreter = tflite.Interpreter(model_path=tfModel)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']

if mfcc is True:
    tensor_spec =(tf.TensorSpec([None,49,10,1], dtype=tf.float32), tf.TensorSpec([None], dtype=tf.int64))
else:
    tensor_spec =(tf.TensorSpec([None,32,32,1], dtype=tf.float32), tf.TensorSpec([None], dtype=tf.int64))

test_ds = tf.data.experimental.load(dataset_dir, tensor_spec) 
test_ds= test_ds.unbatch().batch(1)

accuracy=0
count= 0 
for x, y_true in test_ds: 
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    y_pred = interpreter.get_tensor(output_details[0]['index'])
    y_pred = y_pred.squeeze()
    y_pred = np.argmax(y_pred)
    y_true = y_true.numpy().squeeze()
    accuracy += y_pred == y_true
    count += 1 

accuracy/=float(count)
print("Accuracy {}".format(accuracy*100))



