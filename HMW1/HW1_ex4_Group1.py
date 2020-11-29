import tensorflow as tf
from datetime import datetime 
import csv 
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='./raw_data', help='Directory of the input raw data')
parser.add_argument('--output', type=str, default='./fusion.tfrecord', help='Output tfrecord file')

args, _ = parser.parse_known_args()

inputDir = args.input
outputFile = args.output

  


# CSV file
samples = inputDir + "/samples.csv"

with tf.io.TFRecordWriter(outputFile) as writer:
    with open(samples, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            date = row[0] 
            hour = row[1]
            date_hour = date + "," + hour 
            temp = int(row[2])
            humidity = int(row[3])
            audioFile = str(row[4])
            date_hour = datetime.strptime(date_hour, '%d/%m/%Y,%H:%M:%S')
            posix=datetime.timestamp(date_hour)
            posix_int = int(posix)
            audio = tf.io.read_file(inputDir + "/" + audioFile)
            audio = audio.numpy() 
            
            posix_date_hour_feature = tf.train.Feature(int64_list=tf.train.Int64List(value = [posix_int]))
        
            temp_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[temp]))
            
            humidity_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[humidity])) 
            
            audioFile_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[audio]))
           
            mapping = {'datetime': posix_date_hour_feature,
                        'temperature': temp_feature,
                        'humidity': humidity_feature,
                        'audio': audioFile_feature}
            
            example = tf.train.Example(features=tf.train.Features(feature = mapping))
            
            writer.write(example.SerializeToString())
             


    
