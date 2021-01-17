import subprocess

# performance = ['sudo', 'sh', '-c', 'echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor']
# powersave = ['sudo', 'sh', '-c', 'echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor']

# subprocess.check_call(performance)

import argparse
import numpy as np
import tensorflow.lite as tflite
import tensorflow as tf
import numpy as np
import zlib
import os
import sys

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True, help='Version of the model')

args, _ = parser.parse_known_args()

version = args.version


compressed_tfModel = './{}.tflite'.format(version)
tfModel = compressed_tfModel
dataset_dir = './kws_test_{}'.format(version)
mfcc = True

if version=="big":
    tensor_spec_dimension = [None, 48, 10, 1]
else:
    tensor_spec_dimension = [None, 65, 10, 1]

# Decompress it

interpreter = tflite.Interpreter(model_path=tfModel)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)

input_shape = input_details[0]['shape']

tensor_spec =(tf.TensorSpec(tensor_spec_dimension, dtype=tf.float32), tf.TensorSpec([None], dtype=tf.int64))

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
print(f"Size of decompressed model: {os.path.getsize(tfModel)/1024} kB")
print(f"Size of compressed model: {os.path.getsize(compressed_tfModel)/1024} kB")
print("Accuracy: {}".format(accuracy*100))
