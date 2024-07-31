# import libraries 
import numpy as np
import tensorflow as tf
import os
import time

# get the current working directory
current_path = os.getcwd()

# define the TFLite model name 
trained_model_float = "daefloat"

# define the data files names 
noisy_data_name = "x_test_noisy1.npy"
clean_data_name = "x_test_clean1.npy"

# create full paths to the files 
full_path_model_float = os.path.join(current_path, trained_model_float)
full_path_noisy = os.path.join(current_path, noisy_data_name)
full_path_clean = os.path.join(current_path, clean_data_name)

# load the data 
x_test_noisy = np.load(full_path_noisy)
x_test_clean = np.load(full_path_clean)

# load the TFLite model and allocate tensors  
interpreter = tf.lite.Interpreter(model_path=full_path_model_float)
interpreter.allocate_tensors()
    
# get the input and output tensors 
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
    
# prepare the test dataset
test_data = x_test_noisy.astype(np.float32)
        
total_inference_time = 0
counter = 0

# Run inference on each test sample
results = []

for sample in test_data:
    
    counter = counter + 1
    
    # set input tensor
    interpreter.set_tensor(input_details[0]['index'], sample.reshape((1, 800)))
    
    start_time = time.time()
    # run inference
    interpreter.invoke()
    
    # calculate the elapsed time
    end_time = time.time()
    inference_time = end_time - start_time
    
    total_inference_time = total_inference_time + inference_time
    
    # get the output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results.append(output_data)

# convert the results to a numPy array
results = np.array(results)
results = np.squeeze(results, axis=(1,3))

decoded_layer = results

# print the results and the inference time
print("Total Inference time: {:.2f} seconds".format(total_inference_time))
print("Counter: {:.2f}".format(counter))
print (decoded_layer)