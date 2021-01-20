import subprocess

performance = ['sudo', 'sh', '-c', 'echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor']
powersave = ['sudo', 'sh', '-c', 'echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor']

subprocess.check_call(performance)

from DoSomething import DoSomething
import tensorflow as tf
import numpy as np
from scipy import signal
import datetime
import time
import os
import json
import base64
import sys


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


        if self.resampling_rate is not None:
            rate = self.resampling_rate

        else:
            rate = self.sampling_rate
        
        
        self.num_frames = (rate - self.frame_length) // self.frame_step + 1
        
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

        return mfccs



class CooperativeClient(DoSomething):
    def __init__(self, clientID):
        super().__init__(clientID)
        self.last_layer_client1 = []
        self.last_layer_client2 = []
        self.true_labels = []

    def notify(self, topic, msg):
        
        senml = json.loads(msg)
        output_string = senml["e"][0]["vd"]
        timestamp = senml["bt"]

        output_bytes = base64.b64decode(output_string)
        output = tf.io.decode_raw(output_bytes, tf.float32)

        if senml["bn"] == "inference_client1":
            self.last_layer_client1.append(output)
        elif senml["bn"] == "inference_client2":
            self.last_layer_client2.append(output)
        else:
            print("invalid client: ", senml["bn"])
            sys.exit()



start = time.time()

if __name__ == "__main__":

    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    #Se non uso id diversi per abbonarmi allo stesso topic ho robe strane
    test = CooperativeClient("CooperativeInference")
    test.run()
    test.myMqttClient.mySubscribe("/s276033/output_channel")

    sampling_rate = 16000
    resampling_rate = None
    frame_length = 640
    frame_step = 320

    ROOT_DIR = "./"

    zip_path = tf.keras.utils.get_file(
            origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
            fname='mini_speech_commands.zip',
            extract=True,
            cache_dir='.', cache_subdir='data')

    data_dir = os.path.join('.', 'data', 'mini_speech_commands')
    filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
    filenames = tf.random.shuffle(filenames)

    test_files = tf.strings.split(tf.io.read_file(ROOT_DIR +'kws_test_split.txt'),sep='\n')[:-1]

    LABELS = ['down', 'stop', 'right', 'left', 'up', 'yes', 'no', 'go']

    MFCC_OPTIONS = {'frame_length': frame_length, 'frame_step': frame_step, 'mfcc': True,
        'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40,
        'num_coefficients': 10}

    options = MFCC_OPTIONS

    Preprocess = preprocess(LABELS, sampling_rate=sampling_rate, resampling_rate=resampling_rate, **options)


    accuracy = 0
    count = 0
    for file_path in test_files:

        now = datetime.datetime.now()
        timestamp = int(now.timestamp())


        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2]
        label_id = tf.argmax(label == LABELS)
        audio_binary = tf.io.read_file(file_path)
        mfccs = Preprocess.preprocess_with_mfcc(audio_binary)
        input_tensor = mfccs
        y_true = label_id

        test.true_labels.append(int(y_true))


        audio_b64bytes = base64.b64encode(input_tensor.numpy())
        audio_string = audio_b64bytes.decode()


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
            ],
        }

        body = json.dumps(body)

        test.myMqttClient.myPublish("/s276033/my_prep_audio", body)

        count += 1


    tmp = time.time()
    while len(test.last_layer_client1) != count and len(test.last_layer_client2) != count:
        if (time.time()-tmp > 60):
            break
        time.sleep(0.1)
    #print(time.time()-tmp)


    # wait for things, estimate this time with the immediately above commented code
    #time.sleep(1)

    body = {
        "bn": "http://192.168.1.92/",
        "bt": timestamp,
        "e": [
            {
                "n": "stop",
                "u": "/",
                "t": 0,
                "v": '/'
            }
        ],
    }


    body = json.dumps(body)

    test.myMqttClient.myPublish("/s276033/my_prep_audio", body)

    
    if len(test.true_labels)!=count or len(test.last_layer_client1)!=count or len(test.last_layer_client2)!=count:
        print("Some messages went lost, restart application")
        test.end()
        sys.exit()
    #print(count)


    accuracy = 0
    for i in range(count):
        pred_probs1 = tf.nn.softmax(test.last_layer_client1[i])
        pred_probs2 = tf.nn.softmax(test.last_layer_client2[i])

        pred_probs1 = tf.sort(pred_probs1, direction='DESCENDING')
        pred_probs2 = tf.sort(pred_probs2, direction='DESCENDING')

        score_margin1 = pred_probs1[0] - pred_probs1[1]
        score_margin2 = pred_probs2[0] - pred_probs2[1]

        if score_margin1 > score_margin2:
            prediction = np.argmax(test.last_layer_client1[i])
        else:
            prediction = np.argmax(test.last_layer_client2[i])
            
        accuracy += prediction==test.true_labels[i]

    
    accuracy/=float(count)
    print("Accuracy: {}".format(accuracy*100))   
    print(time.time()-start)

    test.end()