from DoSomething import DoSomething
import tensorflow as tf
import tensorflow.lite as tflite
import numpy as np
import time
import base64
import datetime
import json
import argparse


class InferenceClient(DoSomething):
    def __init__(self, clientID, model_path):
        super().__init__(clientID)
        self.model_path = model_path
        

    #action to do when a message is received
    def notify(self, topic, msg):
        #get the result of the nn and forward to client

        now = datetime.datetime.now()
        timestamp = int(now.timestamp())

        senml = json.loads(msg)

        audio_string = senml["e"][0]["vd"]

        #print(audio_string)

        #decode audio from base64
        audio_bytes = base64.b64decode(audio_string)
        raw_audio = tf.io.decode_raw(audio_bytes, tf.float32)

        input_tensor = np.reshape(raw_audio, (1, 49, 10, 1))
        #print(input_tensor, input_tensor.shape)

        #get the nn from file
        interpreter = tflite.Interpreter(model_path="./HMW3/big.tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        #compute the nn predictin  
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        my_output = interpreter.get_tensor(output_details[0]['index'])
        my_output = my_output.squeeze()

        #print(my_output, type(my_output))


        output_b64bytes = base64.b64encode(my_output)
        output_string = output_b64bytes.decode()

        body = {
            # my url
            "bn": "http://192.168.1.92/",
            "bt": timestamp,
            "e": [
                {
                    "n": "output",
                    "u": "/",
                    "t": 0,
                    "vd": output_string
                }
            ],
        }

        body = json.dumps(body)

        
        #print("Output:", msg)  
        
        #Forward result
        test.myMqttClient.myPublish ("/s276033/output_channel", body) 	

        #print("Result {} forwarded on output_channel".format(msg), self.clientID)
    

#per capire se fare solo funzione o solo classe vedo se devo ridefinire anche il "onNotificationReceived"(probabilmente si )

if __name__ == "__main__":

    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)


    #I take model path and
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='model path')

    args, _ = parser.parse_known_args()

    model_path=args.model

    # Se non uso id diversi per abbonarmi allo stesso topic ho robe strane
    clientID = "inference_client" + model_path.split(".")[0]
    
    #subscribe to prep_audio channel
    test = InferenceClient(clientID, model_path)
    test.run()
    test.myMqttClient.mySubscribe("/s276033/my_prep_audio")
    a = 0

    while True:
        time.sleep(0.01)
    # while (a < 30):
    #     a += 1
    #     time.sleep(1)
    test.end()