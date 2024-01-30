#include <TensorFlowLite.h>

#include "autoencoder_Arduino.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"


const int segment_size = 800;
float input_data[segment_size];
// Define a fixed seed value for the random number generator
const unsigned long fixedSeed = 123;


// Globals 
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

// Create an area of memory to use for input, output, and other TensorFlow arrays.
// You'll need to adjust this by compiling, running, and looking for errors.
constexpr int kTensorArenaSize = 100*1024;
// Keep aligned to 16 bytes for CMSIS
// alignas(16) uint8_t tensor_arena[kTensorArenaSize];
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace



void setup() {

  tflite::InitializeTarget();

  Serial.begin(115200);  // Initialize serial communication
   
  //Load model
  model = tflite::GetModel(autoencoder_Arduino);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;


  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

    // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // x=interpreter->

  
  // input = interpreter->input_tensor(0);
  // output = interpreter->output_tensor(0);


}


void loop() {

  randomSeed(fixedSeed);
  float synthetic_data[800];
  for (int i = 0; i < 800; i++) {
    synthetic_data[i] = random(0, 100);  // Generate random values (adjust range as needed)
  }

 
  // Normalize Data
  int data_length = sizeof(synthetic_data) / sizeof(synthetic_data[0]);
  float min_val = synthetic_data[0];
  float max_val = synthetic_data[0];
  for (int i = 1; i < data_length; i++) {
    if (synthetic_data[i] < min_val) {
      min_val = synthetic_data[i];
    }
    if (synthetic_data[i] > max_val) {
      max_val = synthetic_data[i];
    }
  }

  float normalized_data[800];
  for (int i = 0; i < data_length; i++) {
    normalized_data[i] = (synthetic_data[i] - min_val) / (max_val - min_val);
  }
  
  // Print the generated array
  Serial.begin(115200);
  for (int i = 0; i < 800; i++) {
    Serial.print(normalized_data[i]);
    Serial.print("  ");
  }




  // Model input
  // input->data.f[0] = normalized_data;
  memcpy(input->data.f, normalized_data, input->bytes);
  // for (int i = 0; i < 800; i++) {
  //     input->data.f[i] = normalized_data[i];
  // }

  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed");
    return;
  }

  // interpreter->Invoke();


  // get the output
  float output_data[800];  // Initialize an array for the output data

  // Copy the output tensor data to your output array
  // for (int i = 0; i < 800; i++) {
  //   output_data[i] = output->data.f[i];
  // }
  // output_data = output->data.f[0];
  memcpy(output_data, output->data.f, output->bytes);


  // Print the output
  // Serial.begin(115200);
  // for (int i = 0; i < data_length; i++) {
  //   Serial.print(output_data[i], 2); // Print with 2 decimal places
  //   Serial.print(" "); 
  // }

  // Delay for a while or implement your main program logic
  delay(20000);
}
