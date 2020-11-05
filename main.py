import numpy as np
import tensorflow as tf
import csv



filename = './numbers.tfrecord'

data = 0
ora = 0
temperatura = 0
umidità = 0
audioFile = 0

folder = "./raw_data/"
file = folder + "samples.csv"

with tf.io.TFRecordWriter(filename) as writer:
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            #print(row, type(row))
            row = row[0].split(",")
            #print(row, type(row)) 
            #print()
            date = tf.constant(row[0]).numpy()
            hour = tf.constant(row[1]).numpy()
            temp = int(row[2])
            humidity = int(row[3])
            audioFile = tf.constant(row[4]).numpy()

            # Feature≠Features
            date_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[date]))
            hour_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[hour]))
            temp_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[temp]))
            humidity_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[humidity]))
            audioFile_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[audioFile]))

            mapping = {'date': date_feature,
                        'hour' : hour_feature,
                        'temperature': temp_feature,
                        'humidity': humidity_feature,
                        'audioFile': audioFile_feature}
            
            example = tf.train.Example(features=tf.train.Features(feature=mapping))
            
            print(example)

            writer.write(example.SerializeToString())
            

    