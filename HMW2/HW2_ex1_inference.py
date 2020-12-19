import subprocess

performance = ['sudo', 'sh', '-c', 'echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor']
powersave = ['sudo', 'sh', '-c', 'echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor']

subprocess.check_call(performance)

import argparse
import numpy as np
import tensorflow.lite as tflite
import tensorflow as tf
import zlib
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--tfModel', type=str, default='./Group1_th_a.tflite.zlib', help='Tflite model')
parser.add_argument('--testDs', type=str, default='./th_test', help='Test dataset')


args, _ = parser.parse_known_args()

tfModel = args.tfModel
test_ds = args.testDs

if "zlib" in tfModel:
    print(f"Size of compressed tflite model: {os.path.getsize(tfModel)/1024} kB")

    str_object1 = open(tfModel, 'rb').read()
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

tensor_specs = (tf.TensorSpec([None, 6, 2], dtype=tf.float32), tf.TensorSpec([None, 6, 2]))
test_ds = tf.data.experimental.load(test_ds, tensor_specs)

test_ds = test_ds.unbatch().batch(1)

class MsMoMAE(tf.keras.metrics.Metric):
    def __init__(self, name='MsMoMAE', **kwargs):
        super(MsMoMAE, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros', shape=(2, ))
        self.count = self.add_weight(name='count', initializer='zeros')

    def reset_states(self):
        self.count.assign(tf.zeros_like(self.count))
        self.total.assign(tf.zeros_like(self.total))
        return

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = tf.abs(y_pred - y_true)
        error = tf.reduce_mean(error, axis=[0,1])
        self.total.assign_add(error)
        self.count.assign_add(1)
        return

    def result(self):
        result = tf.math.divide_no_nan(self.total, self.count)
        return result


MaE=MsMoMAE()

for el in test_ds:     
  interpreter.set_tensor(input_details[0]['index'],el[0])
  interpreter.invoke()

  my_output = interpreter.get_tensor(output_details[0]['index']) 
    
  MaE.update_state(my_output,el[1])

print(f"Score for temperature {MaE.result()[0].numpy()}")
print(f"Score for humidity: {MaE.result()[1].numpy()}")


 
