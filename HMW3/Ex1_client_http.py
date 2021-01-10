import tensorflow as tf
import tensorflow.lite as tflite
import base64
import datetime
import requests
from io import BytesIO
import numpy as np
from scipy import signal
import wave
import argparse
import time
import json
import zlib
import sys
import os
import shutil

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

threshold = np.NaN

sampling_rate = 16000
resampling_rate = None
frame_length = 640
frame_step = 320
tensor_spec_dimension = [None, 49, 10, 1]

ROOT_DIR = "./"
tfModel = "./Group1_kws_a.tflite.zlib"
url = "http://192.168.1.232:8080/"

zip_path = tf.keras.utils.get_file(
        origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        fname='mini_speech_commands.zip',
        extract=True,
        cache_dir='.', cache_subdir='data')

data_dir = os.path.join('.', 'data', 'mini_speech_commands')
filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)

test_files = tf.strings.split(tf.io.read_file(ROOT_DIR +'kws_test_split.txt'),sep='\n')[:-1]

#TODO: labels provvisorie per debug, modificare con quelle successive,
LABELS = ['right', 'go', 'no', 'left', 'stop', 'up', 'down', 'yes']
# Labels prof
# LABELS = ['down', 'stop', 'right', 'left', 'up', 'yes', 'no', 'go']


class preprocess:
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

        self.num_frames = (self.sampling_rate - self.frame_length) // self.frame_step + 1


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

    def read(self, audio_bytes):
        audio, _ = tf.audio.decode_wav(audio_bytes)
        audio = tf.squeeze(audio, axis=1)

        if self.resampling_rate is not None:
            audio = tf.numpy_function(self.custom_resampling, [audio], tf.float32)

        return audio


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

    def preprocess_with_stft(self, audio_bytes):
        audio = self.read(audio_bytes)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        spectrogram = tf.image.resize(spectrogram, [32, 32])

        return spectrogram

    def preprocess_with_mfcc(self, audio_bytes):
        audio = self.read(audio_bytes)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        mfccs = self.get_mfccs(spectrogram)
        # Reshaping since only 1 audio at time si given for inference 
        mfccs = tf.reshape(mfccs, [1, self.num_frames, self.num_coefficients, 1])
        #mfccs = tf.expand_dims(mfccs, -1)

        return mfccs


MFCC_OPTIONS = {'frame_length': frame_length, 'frame_step': frame_step, 'mfcc': True,
        'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40,
        'num_coefficients': 10}

options = MFCC_OPTIONS

Preprocess = preprocess(LABELS, sampling_rate=sampling_rate, resampling_rate=resampling_rate, **options)


if ".zlib" in tfModel:
    # Decompress it
    str_object1 = open(tfModel, 'rb').read()
    str_object2 = zlib.decompress(str_object1)
    tfModel = tfModel.replace('.zlib', '')
    f = open(tfModel, 'wb')
    f.write(str_object2)
    f.close()


interpreter = tflite.Interpreter(model_path=tfModel)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']


def success_checker(predictions, threshold):
    """
        Return True if we need to send the audio file to the big model
    """
    predictions = tf.sort(predictions, direction='DESCENDING')
    score_margin = predictions[0] - predictions[1]
    if score_margin > threshold:
        return True
    else:
        return False


accuracy = 0
count = 0
for file_path in test_files:
    parts = tf.strings.split(file_path, os.path.sep)
    label = parts[-2]
    label_id = tf.argmax(label == LABELS)
    #print(file_path, label, label_id)
    audio_binary = tf.io.read_file(file_path)
    mfccs = Preprocess.preprocess_with_mfcc(audio_binary)
    input_tensor = mfccs
    y_true = label_id
    

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    
    y_pred = interpreter.get_tensor(output_details[0]['index'])
    y_pred = y_pred.squeeze()

    BIG = success_checker(y_pred, 3)

    #print(BIG)

    if BIG is True:
        now = datetime.datetime.now()
        timestamp = int(now.timestamp())

        

        #TODO:vedere come transformare un tensore in stringa base64
        audio_bytes = audio_binary.numpy()
        audio_b64bytes = base64.b64encode(audio_bytes)
        audio_string = audio_b64bytes.decode()
        #print(type(audio_string))

        body = {
            # my url
            "bn": "http://192.168.1.92/",
            "bt": timestamp,
            "e": [
                {
                    "n": "audio",
                    "u": "/",
                    "t": 0,
                    "vd": audio_string
                }
            ]
        }

        r = requests.put(url, json=body)

        if r.status_code == 200:
            #print("little: ", np.argmax(y_pred))
            rbody = r.json()
            #TODO: do stuff
            y_pred = int(rbody['predicted_label'])
            #print("Big: ", y_pred, type(y_pred))
            #sys.exit()

        else:
            #TODO: say what error
            print("Error")
            print(r.text)

    

    else:
        y_pred = np.argmax(y_pred)
    
    y_true = y_true.numpy().squeeze()
    
    accuracy += y_pred == y_true
    count += 1

    # if count == 4:
    #     sys.exit()

accuracy/=float(count)
print("Accuracy: {}".format(accuracy*100))

    

    


