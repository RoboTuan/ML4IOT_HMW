from DoSomething import DoSomething
import time
import base64

class InferenceClient(DoSomething,model_path):
    def __init__(self, clientID):
        super().__init__(clientID)
        self.model_path= model_path
        

    #action to do when a message is received
    def notify(self, topic, msg):
        #get the result of the nn and forward to client

        #decode audio from base64
        prep_audio_decoded = base64.b64decode(msg)
        #get the nn from file
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        
        #compute the nn predictin  
        interpreter.set_tensor(input_details[0]['index'], prep_audio_decoded)
        interpreter.invoke()
        my_output = interpreter.get_tensor(output_details[0]['index'])
        
        print("Output:", my_output)  
        
        #Forward result
        test.myMqttClient.myPublish ("/output_channel", output) 	

        print("Result {} forwarded on output_channel".format(result))
       

#per capire se fare solo funzione o solo classe vedo se devo ridefinire anche il "onNotificationReceived"(probabilmente si )

if __name__ == "__main__":

    #I take model path and
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='model path')

    args, _ = parser.parse_known_args()

    model_path=args.model
	
    #subscribe to prep_audio channel
    test = InferenceClient("inference_client",model_path)
    test.run()
    test.myMqttClient.mySubscribe("/my_prep_audio")
    a = 0
    while (a < 30):
		a += 1
		time.sleep(1)
    test.end()