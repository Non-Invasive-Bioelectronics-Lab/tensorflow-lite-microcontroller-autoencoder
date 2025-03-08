# This script loads the Autoencoder_revision model (basic model) and converts it to TFLite models:
# 1- daefloat (float TFLite model)
# 2- daeint8 (int8 TFLite model (quantized))

# include libraries 
import tensorflow as tf
import numpy as np
from tensorflow import keras
import os

# get the current working directory
current_path = os.getcwd()
trained_model_name = "Autoencoder_revision"
full_path_model = os.path.join(current_path, trained_model_name)

# load the DAE model
autoencoder_model = tf.keras.models.load_model(full_path_model)
autoencoder_model.summary()

# convert the model to tflite (float)
converter = tf.lite.TFLiteConverter.from_keras_model(autoencoder_model)
daefloat = converter.convert()

# save the converted model as a file
open("daefloat", "wb").write(daefloat)

# export the converted float model to a header file
# !xxd -i daefloat > daefloat.h

# load the data 

# Full paths to the files
noisy_data_name = "x_test_noisy1.npy"
clean_data_name = "x_test_clean1.npy"
full_path_noisy = os.path.join(current_path, noisy_data_name)
full_path_clean = os.path.join(current_path, clean_data_name) 
x_test_noisy = np.load(full_path_noisy)
x_test_clean = np.load(full_path_clean)

# export the data to text files (optional)
noisy_text_file = "x_test_noisy.txt"
clean_text_file = "x_test_clean.txt"
full_path_noisy_text = os.path.join(current_path, noisy_text_file)
full_path_clean_text = os.path.join(current_path, clean_text_file)
np.savetxt(full_path_noisy_text, x_test_noisy, delimiter=',')
np.savetxt(full_path_clean_text, x_test_clean, delimiter=',')

# convert the x_test_noise data to flaot32 to be used in the representative function 
x_test_noisy = x_test_noisy.astype(np.float32)

# quantize the model to int8 and then convert and export

# we need a representative dataset function
def representative_dataset (num_samples=500):
  for i in range (num_samples):
    yield[x_test_noisy[i].reshape(1, 800)]
    
# convert the model to tflite (int8)
converter = tf.lite.TFLiteConverter.from_keras_model(autoencoder_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.representative_dataset = representative_dataset
daeint8 = converter.convert()

# save the converted model
open("daeint8", "wb").write(daeint8)

# export the converted int8 model to a header file
# !xxd -i daeint8 > daeint8.h
