"""
Extract INT8 TFLite model weights and generate pure-C inference engine.
No TFLite Micro runtime needed on the target device.
"""
import numpy as np
import struct
import os
import json

def read_tflite_model(path):
    """Parse TFLite flatbuffer to extract model structure and weights."""
    # We'll use TFLite interpreter to extract everything
    import tensorflow as tf
    
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input:  {input_details}")
    print(f"Output: {output_details}")
    
    # Get all tensor details
    tensor_details = interpreter.get_tensor_details()
    
    print(f"\n=== All Tensors ({len(tensor_details)}) ===")
    layers = []
    for td in tensor_details:
        name = td['name']
        shape = td['shape']
        dtype = td['dtype']
        quant = td['quantization_parameters']
        
        # Check if it's a constant (weight/bias) tensor
        try:
            data = interpreter.get_tensor(td['index'])
            is_const = True
        except:
            is_const = False
            data = None
        
        print(f"  [{td['index']:3d}] {name:50s} shape={shape} dtype={dtype} "
              f"scales={quant['scales'].shape if len(quant['scales'])>0 else 'none'} "
              f"const={is_const}")
        
        layers.append({
            'index': td['index'],
            'name': name,
            'shape': tuple(shape),
            'dtype': dtype,
            'quantization': quant,
            'data': data,
            'is_const': is_const
        })
    
    return interpreter, layers, input_details, output_details

def extract_layer_info(interpreter):
    """Extract layer-by-layer ops from the TFLite model."""
    # Use the internal op resolver
    from tensorflow.lite.python.interpreter import Interpreter
    
    # Get the op list by examining the flatbuffer
    details = interpreter._get_op_details(0)  # subgraph 0
    return details

def main():
    model_path = "output/model_int8.tflite"
    output_dir = "output"
    
    interpreter, layers, input_det, output_det = read_tflite_model(model_path)
    
    print("\n=== Input Details ===")
    print(f"  Shape: {input_det[0]['shape']}")
    print(f"  Dtype: {input_det[0]['dtype']}")
    print(f"  Scale: {input_det[0]['quantization_parameters']['scales']}")
    print(f"  ZP:    {input_det[0]['quantization_parameters']['zero_points']}")
    
    print("\n=== Output Details ===")
    print(f"  Shape: {output_det[0]['shape']}")
    print(f"  Dtype: {output_det[0]['dtype']}")
    print(f"  Scale: {output_det[0]['quantization_parameters']['scales']}")
    print(f"  ZP:    {output_det[0]['quantization_parameters']['zero_points']}")
    
    # Test inference with a known input
    input_shape = input_det[0]['shape']
    test_input = np.zeros(input_shape, dtype=np.int8)
    test_input[:] = -128  # All zeros after dequant
    
    interpreter.set_tensor(input_det[0]['index'], test_input)
    interpreter.invoke()
    test_output = interpreter.get_tensor(output_det[0]['index'])
    print(f"\n=== Test (all-zero input) ===")
    print(f"  Raw output: {test_output}")
    
    out_scale = output_det[0]['quantization_parameters']['scales'][0]
    out_zp = output_det[0]['quantization_parameters']['zero_points'][0]
    dequant = (test_output.astype(np.float32) - out_zp) * out_scale
    print(f"  Dequantized: {dequant}")
    
    # Now let's extract all weight tensors and generate C code
    print("\n=== Extracting weights for C code generation ===")
    
    weight_tensors = []
    for layer in layers:
        if layer['data'] is not None and len(layer['shape']) > 1:
            print(f"  Weight: {layer['name']} shape={layer['shape']} dtype={layer['dtype']}")
            weight_tensors.append(layer)
    
    # Print model structure summary
    print("\n=== Model Summary ===")
    print(f"  Input:  {input_shape} (INT8, scale={input_det[0]['quantization_parameters']['scales'][0]:.10f}, zp={input_det[0]['quantization_parameters']['zero_points'][0]})")
    print(f"  Output: {output_det[0]['shape']} (INT8, scale={out_scale:.10f}, zp={out_zp})")
    
    for wt in weight_tensors:
        q = wt['quantization']
        print(f"  {wt['name']}: shape={wt['shape']}, "
              f"n_scales={len(q['scales'])}, "
              f"range=[{wt['data'].min()}, {wt['data'].max()}]")

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
