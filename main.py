import numpy as np
import tensorflow as tf


filename = './numbers.tfrecord'
data = 0
ora = 0
temperatura = 0
unidità=0
audioFile=0

with tf.io.TFRecordWriter(filename) as writer:
    for i in range(10):

        # Feature≠Features
        x_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[x[i]]))
        y_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[y[i]]))
        
        mapping = {'integer': x_feature,
                    'float' : y_feature}
        
        example = tf.train.Example(features=tf.train.Features(feature=mapping))
        
        writer.write(example.SerializeToString())
        

    