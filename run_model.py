from pycoral.utils.edgetpu import make_interpreter

tf_lite_model = 'autoencoder_int8.tflite' # model after conversion using % edgetpu_compiler -s autoencoder_int8.tflite


def main():
    interpreter = make_interpreter()
    interpreter.allocate_tensors()
    

if __name__ == '__main__':
  main()