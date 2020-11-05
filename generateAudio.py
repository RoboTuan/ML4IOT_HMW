import argparse
import csv
from datetime import datetime, date, time, timedelta
import time as t
from board import D4
import adafruit_dht

import os
import pyaudio
import wave
import time as t

#TODO:
# modificare python in modo che data e ora siano
# due campi diversi del csv

def main(freq, per, out):
    print(f"Selected frequencyof measurements: {freq} seconds")
    print(f"Selected period of measurements: {per} seconds")
    print(f"Output file: {out}")

    readings = []
    dht_device = adafruit_dht.DHT11(D4)

    t_start = datetime.now().replace(microsecond=0)
    t_end = t_start + timedelta(seconds=per)
    
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 48000
    CHUNK = RATE//10
    RECORD_SECONDS = 1
    #WAVE_OUTPUT_FILENAME = "file.wav"
    counter = 0

    

    while datetime.now().replace(microsecond=0) < t_end:

        temperature = dht_device.temperature
        humidity = dht_device.humidity
        print(f"Temperature: {temperature}, Humidity: {humidity}")
        
        

        print(f"Bith depth: 16")
        print(f"Sample rate: {RATE}")
        print(f"Chunk size: {CHUNK}")
        print(f"Recording time: {RECORD_SECONDS} seconds")
        #print(f"Output file name: {WAVE_OUTPUT_FILENAME}")
        print()

        folder = "./audio/"
        file = "file"+str(counter)+".wav" 
        counter += 1

        WAVE_OUTPUT_FILENAME = folder + file

        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

        print("recording...")

        frames = []

        start_sensing = t.time()
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        end_sensing = t.time()
        sensing_time = end_sensing - start_sensing

        print("finished recording")
        
        # stop Recording
        stream.stop_stream()
        stream.close()
        # Destroy auio object to destroy memory
        audio.terminate()

        t_start_storing = t.time()

        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()

        t_end_storing = t.time()

        print(f"time for sensing the audio: {sensing_time}")
        print(f"Time for storing the data on disk: {round(t_end_storing-t_start_storing, 4)} seconds")

        wav_size = os.path.getsize(WAVE_OUTPUT_FILENAME)
        print(f"The size of the wav file is: {int(wav_size/1024)} KiloBytes")

        readings.append((datetime.now().replace(microsecond=0), temperature, humidity, file))
                
        t.sleep(freq)

    with open(out, "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for line in readings:
            writer.writerow(line)
    
        
                   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frequency', type=int, default='3', help='Frequency of measurements')
    parser.add_argument('--period', type=int, default='20', help='Period of measurements')
    parser.add_argument('--output', type=str, default='resultsLab01Ex4.csv', help='Output filename')

    args, _ = parser.parse_known_args()

    freq = args.frequency
    per = args.period
    out = args.output

    main(freq, per, out)
