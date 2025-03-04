import os
import numpy as np
import tensorflow as tf

# Paths to the models
float_model_path = "daefloat"  # Path to the float32 model
int8_model_path = "daeint8"    # Path to the int8 model

# Function to analyze the memory usage of the models in bytes, kilobytes, and megabytes
def analyze_model_size(model_path):
    model_size_bytes = os.path.getsize(model_path)  # Get model size in bytes
    model_size_kb = model_size_bytes / 1024        # Convert to kilobytes
    model_size_mb = model_size_kb / 1024          # Convert to megabytes
    return model_size_bytes, model_size_kb, model_size_mb

# Analyze memory usage for both models
float_model_size_bytes, float_model_size_kb, float_model_size_mb = analyze_model_size(float_model_path)
int8_model_size_bytes, int8_model_size_kb, int8_model_size_mb = analyze_model_size(int8_model_path)

# Print results for float32 and int8 models
print("Model Analysis Report:")

# Float32 model analysis
print(f"\nFloat32 Model Analysis:")
print(f"  - Model Size: {float_model_size_bytes} Bytes")
print(f"  - Model Size: {float_model_size_kb:.2f} KB")
print(f"  - Model Size: {float_model_size_mb:.2f} MB")

# Int8 model analysis
print(f"\nInt8 Model Analysis:")
print(f"  - Model Size: {int8_model_size_bytes} Bytes")
print(f"  - Model Size: {int8_model_size_kb:.2f} KB")
print(f"  - Model Size: {int8_model_size_mb:.2f} MB")

# Size reduction comparison
print(f"\nSize Reduction (int8 vs float32): {100 * (float_model_size_mb - int8_model_size_mb) / float_model_size_mb:.2f}%")

# Display the memory comparison
print(f"\nMemory Comparison:")
print(f"  - Float32 Model Size: {float_model_size_bytes} Bytes")
print(f"  - Int8 Model Size: {int8_model_size_bytes} Bytes")
print(f"  - Memory Reduction: {100 * (float_model_size_mb - int8_model_size_mb) / float_model_size_mb:.2f}%")
