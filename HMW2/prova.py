import tensorflow as tf 
from scipy import signal 


audio_binary = tf.io.read_file("./data/mini_speech_commands/down/0a9f9af7_nohash_0.wav")
audio, _ = tf.audio.decode_wav(audio_binary)
print(audio)
audio=signal.resample(audio.numpy(),1,2)
audio=tf.squeeze(audio,axis=1)
print(audio)

