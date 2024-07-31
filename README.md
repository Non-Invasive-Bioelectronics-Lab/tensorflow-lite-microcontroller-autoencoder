# tensorflow-lite-microcontroller-autoencoder

Hardware:    
- 'Arduino Nano 33 BLE Sense' OR 'Arduino Nano 33 BLE Sense Rev2'    
- Coral Dev Micro     
- Coral Dev Mini       

Running deep learning auto-encode EEG cleaning model on embedded systems.                 

## Results:

### Arduino Nano:
- Runtime: single inference time (invoke method) took 1,342 milli seconds.       
- Power measurements:      
VBUS = 5.13 V      
IBUS = 0.0240 A = 24 mA      
PBUS = 0.123 W = 123 mW   

### Coral Dev Micro:
- Runtime: single inference time (invoke method) took 273 milli seconds.                 
- Power measurements:      
VBUS = 5.04 V      
IBUS = 0.128 A = 128 mA      
PBUS = 0.648 W = 648 mW   

### Coral Mini Dev:

- Running TFlite Float model on Coral mini Dev:       
Total runtime (1712 samples) = 28.32 seconds      
Single invoke runtime = 16.54 milli seconds       
Power measurements:     
VBUS= 4.97 V   
IBUS = 0.385 A  
PBUS= 1.915 W    

- Running TFLite Int8 model on Coral Mini Dev:     
Total runtime (1712 samples) = 15.28 seconds     
Single invoke runtime = 8.925 milli seconds            
Power measurements:     
VBUS= 4.98 V    
IBUS = 0.351 A    
PBUS=  1.75 W  
