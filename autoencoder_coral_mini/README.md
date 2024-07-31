# Running TFLite model on Coral Dev Mini board     

## Introduction   
In this demo, we explain how to run ML models on embedded systems:      

1- Coral Dev Board Mini   

![Alt text](images/Coral_mini_text.png)

## Running models on Coral Dev mini

After opening a shell to the Coral Dev Mini board, git clone the repository: https://github.com/MSH19/ML_testing. Or, download these files to a folder:

- run_tflite_float_only.py (for running the float TFLite model and printing the results (tested))
- run_tflite_float.py (for running the float TFLite model and printing the evaluation metrics (not tested))
- run_tflite_int8.py (for running the TFLite int8 model and printing the results (tested))
- run_tflite_int8_only.py (for running the float TFLite model and printing the evaluation metrics (not tested))
- daefoat is the TFLite float model    
- daeint8 is the TFLite quantized model

## Run the model on the Coral Dev Board Coral_mini_text
1- Connect the board to a Linux-based PC through the data cable only    
2- Use a USB voltage and current detection meter (such as FNB58 USB tester) to monitor the current through the USB cable (optional)         
3- Wait for the board LED to show green color indicating the that the boot process was completed successfully     
4- Open terminal and run: mdt devices (This should return the name and IP of the connected board)      
5- Use the command "mdt shell" to open a shell on the board using the MDT tool (Managed Device Tool). This starts an interactive shell session on the target device, allowing you to execute commands directly on the board.     
6- Use the command "nmtui" to open a GUI for configuring the internet connection on the board.        
7- Download the GitHub repo to execute the code (use git clone)     

## Results

- Running TFlite Float model on Coral mini Dev:       
Total runtime (1712 samples) = 28.32 seconds      
Single invoke runtime = 16.54 msec       
Power measurements:     
VBUS= 4.97 V   
IBUS = 0.385 A  
PBUS= 1.915 W    

- Running TFLite Int8 model on Coral Mini Dev:     
Total runtime (1712 samples) = 15.28 seconds     
Single invoke runtime = 8.925 msec      
Power measurements:     
VBUS= 4.98 V    
IBUS = 0.351 A    
PBUS=  1.75 W    

- A snapshot of the measurement process is shown below.
- For more details, please refer to the Results.xlsx document.

![Alt text](images/Measurement.PNG)

## edgeTPU compilation:   

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -    

echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list     

sudo apt-get update     

sudo apt-get install edgetpu-compiler     

sudo edgetpu_compiler daeint8     

Edge TPU Compiler version 16.0.384591198      
Input: daeint8       
Output: daeint8_edgetpu.tflite
