# Running the basic auto-encoder model and TFLite models on PC           

- The Autoencoder_revision folder contains the original trained model on PC using Python.                    

- To convert the model to TFLite models, use the script named tflite_conversion. The script also loads the noisy (texting) and clean (reference) data and exports them to text files (optional).       

- The run_basic.py script runs the original auto-encoder model and prints the evaluation metrics.       

- The run_tflite_float.py script runs the TFLite (float) converted model and prints the evaluation metrics.   

- The run_tflite_float_only.py script runs the TFLite (float) converted model and prints result only without metrics.                

- The run_tflite_int8.py script runs the TFLite (int8) converted model and prints the evaluation metrics.  

- The run_tflite_int8_only.py script runs the TFLite (int8) converted model and prints result only without metrics.        

- To convert the TFLite models to header files, use the commands included in the convert_to_header.ipynb file. These header files are required for running the models on the Arduino and Coral Dev Micro boards.     
