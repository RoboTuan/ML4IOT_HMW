import tensorflow as tf
import tensorflow.lite as tflite
import cherrypy 
import json
import base64
from cherrypy.process.wspbus import ChannelFailures
import numpy as np
import tensorflow as tf
import sys


class BigModel(object):
    exposed = True

    def __init__(self):
        big_model_path = "./Group1_kws_a.tflite"

        self.interpreter = tflite.Interpreter(model_path=big_model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()


        self.sampling_rate = 16000

        self.frame_length = 640
        self.frame_step = 320

        self.lower_frequency = 20
        self.upper_frequency = 4000

        self.num_mel_bins = 40
        self.num_coefficients = 10

        num_spectrogram_bins = (self.frame_length) // 2 + 1

        self.num_frames = (self.sampling_rate - self.frame_length) // self.frame_step + 1


        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                self.num_mel_bins, num_spectrogram_bins, self.sampling_rate,
                self.lower_frequency, self.upper_frequency)



    def preprocess(self, audio_bytes):
            # decode and normalize
            audio, _ = tf.audio.decode_wav(audio_bytes)
            audio = tf.squeeze(audio, axis=1)

            # padding
            zero_padding = tf.zeros([self.sampling_rate] - tf.shape(audio), dtype=tf.float32)
            audio = tf.concat([audio, zero_padding], 0)
            audio.set_shape([self.sampling_rate])

            # stft
            stft = tf.signal.stft(audio, frame_length=self.frame_length,
                    frame_step=self.frame_step, fft_length=self.frame_length)
            spectrogram = tf.abs(stft)

            # mfccs
            mel_spectrogram = tf.tensordot(spectrogram,
                    self.linear_to_mel_weight_matrix, 1)
            log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
            mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
            mfccs = mfccs[..., :self.num_coefficients]

            # Reshaping since only 1 audio at time si given for inference 
            mfccs = tf.reshape(mfccs, [1, self.num_frames, self.num_coefficients, 1])
            #mfccs = tf.expand_dims(mfccs, -1)

            return mfccs


    def PUT(self, *path, **query):
        input_body = cherrypy.request.body.read()
        input_body = json.loads(input_body)
        events = input_body['e']

        # vedere se servce
        for event in events:
            if event['n'] == 'audio':
                audio_string = event['vd']

        if audio_string is None:
            raise cherrypy.HTTPError(400, "No audio event")
        
        audio_bytes = base64.b64decode(audio_string)
        mfccs = self.preprocess(audio_bytes)

        input_tensor = mfccs

        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        
        y_pred = self.interpreter.get_tensor(self.output_details[0]['index'])
        y_pred = y_pred.squeeze()
        y_pred = np.argmax(y_pred)

        #print("Big script:", y_pred)

        output_body = {
            'predicted_label': str(y_pred)
        }

        output_body = json.dumps(output_body)

        return output_body




if __name__ == '__main__':
    conf = {
        '/': {
            'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
            #'tools.sessions.on': True
        }
    }
    cherrypy.tree.mount(BigModel(), '/', conf)

    cherrypy.config.update({'server.socket_host': '0.0.0.0'})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start()
    cherrypy.engine.block()