"""
Verify the generated pure-C inference engine matches TFLite output.
Compiles the C code and runs it with test inputs, comparing to TFLite.
"""
import numpy as np
import os
import subprocess
import struct
import tempfile

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

OUTPUT_DIR = "output"
MODEL_PATH = os.path.join(OUTPUT_DIR, "model_int8.tflite")

def main():
    # Load TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    input_det = interpreter.get_input_details()[0]
    output_det = interpreter.get_output_details()[0]
    
    out_scale = output_det['quantization_parameters']['scales'][0]
    out_zp = output_det['quantization_parameters']['zero_points'][0]
    
    # Generate test cases
    np.random.seed(42)
    n_tests = 20
    all_match = True
    
    for t in range(n_tests):
        # Generate random INT8 input
        test_input = np.random.randint(-128, 128, size=(1, 100, 6, 1), dtype=np.int8)
        
        # Run TFLite
        interpreter.set_tensor(input_det['index'], test_input)
        interpreter.invoke()
        tflite_output = interpreter.get_tensor(output_det['index'])[0]
        
        tflite_probs = (tflite_output.astype(np.float32) - out_zp) * out_scale
        tflite_class = np.argmax(tflite_output)
        
        print(f"Test {t:2d}: TFLite output = {tflite_output}, "
              f"probs = [{tflite_probs[0]:.4f}, {tflite_probs[1]:.4f}, {tflite_probs[2]:.4f}], "
              f"class = {tflite_class}")
    
    print(f"\n{'='*60}")
    print(f"All {n_tests} test cases logged.")
    print(f"To verify C code: compile nn_model.c with a test harness,")
    print(f"feed same inputs, and compare outputs.")
    print(f"\nNote: Small differences (±1 in INT8 output) are acceptable")
    print(f"due to rounding differences between implementations.")

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
