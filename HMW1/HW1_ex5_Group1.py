import subprocess

performance = ['sudo', 'sh', '-c', 'echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor']
powersave = ['sudo', 'sh', '-c', 'echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor']
current_freq = ['cat', '/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq']
reset = ['sudo', 'sh','-c', 'echo 1 > /sys/devices/system/cpu/cpufreq/policy0/stats/reset']
statistics = ['cat', '/sys/devices/system/cpu/cpufreq/policy0/stats/time_in_state']

# With call insteam of Popen we wait to finish before continuing
# with the script, it returns 0 for success
# With cech_call, if the exit code was 0 then return, 
# otherwise raise CalledProcessError
subprocess.check_call(powersave)

import pyaudio
import time as t
import io
import tensorflow as tf
import os
from scipy import signal
import numpy as np
import argparse


parser = argparse.ArgumentParser()
# Any internal - characters will be converted to _ characters
# to make sure the string is a valid attribute name.
parser.add_argument('--num-samples', type=int, default=5, help='Number of audio samples')
parser.add_argument('--output', type=str, default='./out', help='Output tfrecord file')

args, _ = parser.parse_known_args()

samples = args.num_samples
outputDir = args.output

# Create directory
if not os.path.isdir(outputDir):
    os.mkdir(outputDir)


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = RATE//10
RECORD_SECONDS = 1

resample_rate = 16000

frame_length = int(16000 * 0.04)
frame_step = int(16000 * 0.02)


audio = pyaudio.PyAudio()

# Stop otherwise it will start
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True, start=False,
                    frames_per_buffer=CHUNK)

# Stop otherwise it will start
#stream.stop_stream()


lower_frequency = 20
upper_frequency = 4000
num_mel_bins = 40
sampling_rate = 16000
num_spectrogram_bins = 321

# Matrix for mfccs
linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                    num_mel_bins,
                    num_spectrogram_bins,
                    sampling_rate,
                    lower_frequency,
                    upper_frequency)


subprocess.Popen(reset)

for sample in range(samples):
    
    # Recording
    t_start = t.time()
    
    stream.start_stream()

    subprocess.Popen(powersave)

    frames_io = io.BytesIO()

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        if i == 9:
            subprocess.Popen(performance)

        data = stream.read(CHUNK)
        frames_io.write(data)

    stream.stop_stream()
    #t_record = t.time()

    # Resampling
    frames_io_buf = frames_io.getvalue()
    frame = np.frombuffer(frames_io_buf, dtype=np.uint16)
    frames_io.close()
    
    audio2 = signal.resample_poly(frame, resample_rate, RATE)
    audio2 = audio2.astype(np.uint16)
    tf_audio = tf.convert_to_tensor(audio2, dtype=tf.float32)
    #t_resample = t.time()
    

    # STFT    
    stft = tf.signal.stft(tf_audio, frame_length=frame_length,
                        frame_step=frame_step,
                        fft_length=frame_length)

    spectrogram = tf.abs(stft)
    #t_stft = t.time()

    

    # MFCCS
    mel_spectrogram = tf.tensordot( spectrogram,
                    linear_to_mel_weight_matrix,
                    1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :10]
    #t_mfccs = t.time()


    # Saving
    f_res = outputDir+ "/" + "mfcc" + str(sample) + ".bin"
    mfccs_ser = tf.io.serialize_tensor(mfccs)
    tf.io.write_file(f_res, mfccs_ser)
    t_savefile = t.time()

    print(round((t_savefile - t_start)*1000, 2))

subprocess.Popen(powersave)

stream.close()
audio.terminate()    

subprocess.Popen(statistics)


