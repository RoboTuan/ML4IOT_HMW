import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import tensorflow.lite as tflite
import zlib
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True, help='Version of the model')

args, _ = parser.parse_known_args()

version = args.version


seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True,
    cache_dir='.', cache_subdir='data')
csv_path, _ = os.path.splitext(zip_path)
df = pd.read_csv(csv_path)

column_indices = [2, 5]
columns = df.columns[column_indices]
data = df[columns].values.astype(np.float32)

n = len(data)
train_data = data[0:int(n*0.7)]
val_data = data[int(n*0.7):int(n*0.9)]
test_data = data[int(n*0.9):]

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

input_width = 6
LABEL_OPTIONS = 2

MODEL_OPTIONS = "MLP"
ALPHA = 0.05

units=6*2


class WindowGenerator:
    def __init__(self, input_width, label_options, mean, std):
        self.input_width = input_width
        self.label_options = label_options
        self.mean = tf.reshape(tf.convert_to_tensor(mean), [1, 1, 2])
        self.std = tf.reshape(tf.convert_to_tensor(std), [1, 1, 2])

    def split_window(self, features):
        input_indeces = np.arange(self.input_width)
        inputs = features[:, :-6, :]
            
        labels = features[:, -6:, :]
        num_labels = 2

        inputs.set_shape([None, self.input_width, 2])
        # vedere se funge
        labels.set_shape([None, self.input_width, num_labels])

        return inputs, labels

    def normalize(self, features):
        features = (features - self.mean) / (self.std + 1.e-6)

        return features

    def preprocess(self, features):
        inputs, labels = self.split_window(features)
        inputs = self.normalize(inputs)

        return inputs, labels

    def make_dataset(self, data, train):
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data,
                # Targets None because we have the targets incorporated in the dataset
                targets=None,
                sequence_length=input_width+6,
                sequence_stride=1,
                batch_size=32)
        ds = ds.map(self.preprocess)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds


generator = WindowGenerator(input_width, LABEL_OPTIONS, mean, std)
train_ds = generator.make_dataset(train_data, True)
val_ds = generator.make_dataset(val_data, False)
test_ds = generator.make_dataset(test_data, False)

# tf.data.experimental.save(train_ds, './th_train')
# tf.data.experimental.save(val_ds, './th_val')
# tf.data.experimental.save(test_ds, './th_test')


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
        #error = tf.reduce_mean(error, axis=1)
        #error = tf.reduce_mean(error, axis=0)
        error = tf.reduce_mean(error, axis=[0,1])
        self.total.assign_add(error)
        self.count.assign_add(1)

        return

    def result(self):
        result = tf.math.divide_no_nan(self.total, self.count)

        return result



if MODEL_OPTIONS == "MLP":
    model = keras.Sequential([
    keras.layers.Flatten(input_shape=(6, 2)),
    keras.layers.Dense(units=int(128*ALPHA), activation='relu', name='first_dense'),
    keras.layers.Dense(units=int(128*ALPHA), activation='relu', name='second_dense'),
    keras.layers.Dense(units, name='third_dense'),
    keras.layers.Reshape([6, 2])
    ])


elif MODEL_OPTIONS == "CNN-1D":
    model = keras.Sequential([
        keras.layers.Conv1D(filters=int(64*ALPHA), kernel_size=3, activation='relu', name='first_conv'),
        keras.layers.Flatten(input_shape=(64,)),
        keras.layers.Dense(units=int(64*ALPHA), activation='relu', name='first_dense'),
        keras.layers.Dense(units, name='second_dense'),
        keras.layers.Reshape([6, 2])
    ])


else:
    print("Invalid model selected")
    sys.exit()


saved_model_dir = './models/climate_{}_{}'.format('normal', str(LABEL_OPTIONS))

model.compile(optimizer='adam',
            loss=[tf.keras.losses.MeanSquaredError()],
            metrics=[MsMoMAE()])


print("Fit model on training data")

history = model.fit(
    train_ds,
    batch_size=32,
    epochs=20,
    validation_data=(val_ds),
)

print("Evaluate on test data")
loss, error = model.evaluate(test_ds, verbose=2)
print(error)

#model.summary()

run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, 6, 2],
tf.float32))
model.save(saved_model_dir, signatures=concrete_func)

#model.save(saved_model_dir)

converter = tf.lite.TFLiteConverter.from_saved_model("./models/climate_normal_2/")

#def representative_dataset_gen():
#    for x, _ in train_ds.take(1000):
#        yield [x]

converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.representative_dataset = representative_dataset_gen

converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

open('./models/normal','wb').write(tflite_model)

with open("./models/normal.zlib", 'wb') as fp:
    tflite_compressed = zlib.compress(tflite_model)
    fp.write(tflite_compressed)

print(f"Size of tflite model: {os.path.getsize('./models/normal')} bytes")
print(f"Size of compressed tflite model: {os.path.getsize('./models/normal.zlib')} bytes")



str_object1 = open('./models/normal.zlib', 'rb').read()
str_object2 = zlib.decompress(str_object1)
f = open('./models/normal_decompressed', 'wb')
f.write(str_object2)
f.close()



interpreter = tflite.Interpreter(model_path="./models/normal_decompressed")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#print(input_details)

input_shape = input_details[0]['shape']


tensor_specs = (tf.TensorSpec([None, 6, 2], dtype=tf.float32), tf.TensorSpec([None, 6, 2]))
#train_ds = tf.data.experimental.load('./th_train', tensor_specs)
#val_ds = tf.data.experimental.load('./th_val', tensor_specs)
test_ds = tf.data.experimental.load('./th_test', tensor_specs)

test_ds=test_ds.unbatch().batch(1)



MaE=MsMoMAE()
for el in test_ds:     
  interpreter.set_tensor(input_details[0]['index'],el[0])
  interpreter.invoke()

  my_output = interpreter.get_tensor(output_details[0]['index']) 
    
  MaE.update_state(my_output,el[1])
print(MaE.result())
