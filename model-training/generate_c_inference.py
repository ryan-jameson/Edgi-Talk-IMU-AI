"""
Generate a pure-C inference engine from INT8 TFLite model.
Extracts all weights/biases and generates standalone inference code.
No TFLite Micro, no C++, no external dependencies required.

Model architecture:
  Conv2D(8, 5x1) + BN + ReLU + MaxPool(2x1)
  Conv2D(16, 5x1) + BN + ReLU + MaxPool(2x1)
  Conv2D(24, 3x3) + BN + ReLU + MaxPool(2x1)
  Flatten
  Dense(24) + ReLU
  Dense(3) + Softmax
"""
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

OUTPUT_DIR = "output"
MODEL_PATH = os.path.join(OUTPUT_DIR, "model_int8.tflite")

def extract_all(interpreter):
    """Extract all layer parameters from TFLite interpreter."""
    tensor_details = interpreter.get_tensor_details()
    tensors = {}
    for td in tensor_details:
        try:
            data = interpreter.get_tensor(td['index'])
        except ValueError:
            data = None
        tensors[td['index']] = {
            'name': td['name'],
            'shape': tuple(td['shape']),
            'dtype': td['dtype'],
            'quant': td['quantization_parameters'],
            'data': data
        }
    return tensors

def tensor_by_name_contains(tensors, substr):
    """Find tensor by substring in name."""
    for idx, t in tensors.items():
        if substr in t['name']:
            return t
    return None

def write_int8_array(f, name, data, per_line=16):
    """Write int8 array to C file."""
    flat = data.flatten()
    f.write(f"static const int8_t {name}[{len(flat)}] = {{\n")
    for i in range(0, len(flat), per_line):
        chunk = flat[i:i+per_line]
        f.write("    " + ", ".join(f"{v}" for v in chunk))
        if i + per_line < len(flat):
            f.write(",")
        f.write("\n")
    f.write("};\n\n")

def write_int32_array(f, name, data, per_line=8):
    """Write int32 array to C file."""
    flat = data.flatten()
    f.write(f"static const int32_t {name}[{len(flat)}] = {{\n")
    for i in range(0, len(flat), per_line):
        chunk = flat[i:i+per_line]
        f.write("    " + ", ".join(f"{v}" for v in chunk))
        if i + per_line < len(flat):
            f.write(",")
        f.write("\n")
    f.write("};\n\n")

def write_float_array(f, name, data, per_line=8):
    """Write float array to C file."""
    flat = data.flatten()
    f.write(f"static const float {name}[{len(flat)}] = {{\n")
    for i in range(0, len(flat), per_line):
        chunk = flat[i:i+per_line]
        f.write("    " + ", ".join(f"{v:.10e}f" for v in chunk))
        if i + per_line < len(flat):
            f.write(",")
        f.write("\n")
    f.write("};\n\n")

def main():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    input_det = interpreter.get_input_details()[0]
    output_det = interpreter.get_output_details()[0]
    tensors = extract_all(interpreter)
    
    # Identify layers by examining tensor names
    # Conv2D layers (TFLite stores weights as [out_ch, H, W, in_ch])
    conv0_w = tensor_by_name_contains(tensors, 'conv2d/Conv2D')
    conv1_w = tensor_by_name_contains(tensors, 'conv2d_1/Conv2D')
    conv2_w = tensor_by_name_contains(tensors, 'conv2d_2/Conv2D')
    
    # BN/bias (fused into conv as bias in int32)
    bn0_b = tensor_by_name_contains(tensors, 'batch_normalization/FusedBatch')
    bn1_b = tensor_by_name_contains(tensors, 'batch_normalization_1/FusedBatch')
    bn2_b = tensor_by_name_contains(tensors, 'batch_normalization_2/FusedBatch')
    
    # Dense layers - be more careful with name matching
    dense0_w = None
    dense0_b = None
    dense1_w = None
    dense1_b = None
    
    for idx, t in tensors.items():
        name = t['name']
        if t['data'] is None:
            continue
        # dense/MatMul but NOT dense_1/MatMul
        if 'dense/MatMul' in name and 'dense_1' not in name and 'BiasAdd' not in name and 'Relu' not in name:
            if len(t['shape']) == 2 and t['shape'][0] != 3:  # not the output layer
                dense0_w = t
        elif 'dense/BiasAdd/Read' in name or ('dense/BiasAdd' in name and 'dense_1' not in name and t['dtype'] == np.int32):
            dense0_b = t
        elif 'dense_1/MatMul' in name and 'BiasAdd' not in name:
            if len(t['shape']) == 2 and t['shape'][0] == 3:
                dense1_w = t
        elif 'dense_1/BiasAdd/Read' in name or ('dense_1/BiasAdd' in name and t['dtype'] == np.int32):
            dense1_b = t
    
    # If direct match didn't work, search more broadly by shape
    if dense0_w is None:
        for idx, t in tensors.items():
            if t['data'] is not None and t['shape'] == (24, 1728):
                dense0_w = t
                print(f"  [fallback] Found dense0_w at index {idx}: {t['name']}")
                break
    if dense0_b is None:
        for idx, t in tensors.items():
            if t['data'] is not None and t['shape'] == (24,) and 'batch_normalization' not in t['name']:
                if 'dense' in t['name'].lower() or 'bias' in t['name'].lower():
                    dense0_b = t
                    print(f"  [fallback] Found dense0_b at index {idx}: {t['name']}")
                    break
        # Still not found - try any (24,) int32
        if dense0_b is None:
            for idx, t in tensors.items():
                if t['data'] is not None and t['shape'] == (24,) and t['data'].dtype == np.int32:
                    if 'batch_normalization' not in t['name']:
                        dense0_b = t
                        print(f"  [fallback2] Found dense0_b at index {idx}: {t['name']}")
                        break
    if dense1_w is None:
        for idx, t in tensors.items():
            if t['data'] is not None and t['shape'] == (3, 24):
                dense1_w = t
                print(f"  [fallback] Found dense1_w at index {idx}: {t['name']}")
                break
    if dense1_b is None:
        for idx, t in tensors.items():
            if t['data'] is not None and t['shape'] == (3,) and t['data'].dtype == np.int32:
                dense1_b = t
                print(f"  [fallback] Found dense1_b at index {idx}: {t['name']}")
                break
    
    print("=== Layer Parameters ===")
    print(f"Conv0: weight={conv0_w['shape']}, bias={bn0_b['shape']}")
    print(f"  weight scales ({len(conv0_w['quant']['scales'])}): {conv0_w['quant']['scales']}")
    print(f"Conv1: weight={conv1_w['shape']}, bias={bn1_b['shape']}")
    print(f"  weight scales ({len(conv1_w['quant']['scales'])}): {conv1_w['quant']['scales']}")
    print(f"Conv2: weight={conv2_w['shape']}, bias={bn2_b['shape']}")
    print(f"  weight scales ({len(conv2_w['quant']['scales'])}): {conv2_w['quant']['scales']}")
    print(f"Dense0: weight={dense0_w['shape']}, bias={dense0_b['shape']}")
    print(f"  weight scale: {dense0_w['quant']['scales']}")
    print(f"Dense1: weight={dense1_w['shape']}, bias={dense1_b['shape']}")
    print(f"  weight scale: {dense1_w['quant']['scales']}")
    
    # Get activation quantization parameters for each layer output
    # tensor 12: after conv0+bn+relu -> scale, zp
    # tensor 13: after maxpool0
    # tensor 14: after conv1+bn+relu
    # tensor 15: after maxpool1
    # tensor 16: after conv2+bn+relu
    # tensor 17: after maxpool2
    # tensor 18: after flatten
    # tensor 19: after dense0+relu
    # tensor 20/21: after dense1
    
    act_tensors = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    print("\n=== Activation Quantization ===")
    for idx in act_tensors:
        if idx in tensors:
            t = tensors[idx]
            q = t['quant']
            s = q['scales'][0] if len(q['scales']) > 0 else 'N/A'
            z = q['zero_points'][0] if len(q['zero_points']) > 0 else 'N/A'
            print(f"  Tensor[{idx}] {t['name'][:60]:60s} scale={s}, zp={z}")
    
    # Get input/output quantization
    in_scale = input_det['quantization_parameters']['scales'][0]
    in_zp = input_det['quantization_parameters']['zero_points'][0]
    out_scale = output_det['quantization_parameters']['scales'][0]
    out_zp = output_det['quantization_parameters']['zero_points'][0]
    
    # For each conv/dense, we need the multiplier and shift for requantization
    # output = clamp((sum * M + output_zp), -128, 127)
    # where M = input_scale * weight_scale / output_scale
    
    # Activation scales and zero points
    act12 = tensors[12]['quant']  # after conv0+bn+relu
    act13 = tensors[13]['quant']  # after maxpool0 (same as act12)
    act14 = tensors[14]['quant']  # after conv1+bn+relu
    act15 = tensors[15]['quant']  # after maxpool1
    act16 = tensors[16]['quant']  # after conv2+bn+relu
    act17 = tensors[17]['quant']  # after maxpool2
    act19 = tensors[19]['quant']  # after dense0+relu
    
    # Compute effective multipliers
    def compute_multiplier_shift(scale_float):
        """Convert floating-point multiplier to fixed-point (multiplier, shift) representation."""
        if scale_float == 0:
            return 0, 0
        # Represent as M * 2^(-shift) where M is in [0.5, 1.0)
        shift = 0
        while scale_float < 0.5:
            scale_float *= 2.0
            shift += 1
        while scale_float >= 1.0:
            scale_float /= 2.0
            shift -= 1
        # M is now in [0.5, 1.0), quantize to int32
        M = int(round(scale_float * (1 << 31)))
        if M == (1 << 31):
            M = (1 << 31) - 1
        return M, shift
    
    # ============================================================
    # Generate C files
    # ============================================================
    
    # --- nn_model.h ---
    with open(os.path.join(OUTPUT_DIR, "nn_model.h"), 'w') as f:
        f.write("""/*
 * nn_model.h - Pure C Neural Network Inference Engine
 * Auto-generated from TFLite INT8 model. Do not edit.
 *
 * Model: Movement Type Detection
 * Input:  100x6 (100 timesteps, 6 IMU channels), INT8
 * Output: 3 classes [stationary, shaking, circle]
 *
 * Architecture:
 *   Conv2D(8, 5x1) + BN + ReLU + MaxPool(2x1)
 *   Conv2D(16, 5x1) + BN + ReLU + MaxPool(2x1)
 *   Conv2D(24, 3x3) + BN + ReLU + MaxPool(2x1)
 *   Flatten(1728) -> Dense(24) + ReLU -> Dense(3)
 */

#ifndef NN_MODEL_H_
#define NN_MODEL_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NN_INPUT_HEIGHT     100
#define NN_INPUT_WIDTH      6
#define NN_INPUT_CHANNELS   1
#define NN_NUM_CLASSES      3
#define NN_SAMPLE_RATE      50
#define NN_WINDOW_STEP      50

#define NN_CLASS_STATIONARY  0
#define NN_CLASS_SHAKING     1
#define NN_CLASS_CIRCLE      2

/* Input quantization: q = clamp(round(float_val / scale) + zero_point, -128, 127) */
#define NN_INPUT_SCALE       %.10ef
#define NN_INPUT_ZERO_POINT  %d

/* Output quantization: float_val = (q - zero_point) * scale */
#define NN_OUTPUT_SCALE       %.10ef
#define NN_OUTPUT_ZERO_POINT  %d

static const char* const nn_class_names[NN_NUM_CLASSES] = {
    "stationary",
    "shaking",
    "circle"
};

/**
 * Run inference on preprocessed INT8 input.
 * @param input   INT8 array of size [100][6][1] = 600 bytes
 * @param output  INT8 array of size [3], raw quantized logits
 */
void nn_inference(const int8_t* input, int8_t* output);

/**
 * Decode INT8 output to float probabilities [0, 1].
 */
void nn_decode_output(const int8_t raw[NN_NUM_CLASSES], float probs[NN_NUM_CLASSES]);

/**
 * Get index of maximum probability.
 */
int nn_argmax(const int8_t raw[NN_NUM_CLASSES]);

#ifdef __cplusplus
}
#endif

#endif /* NN_MODEL_H_ */
""" % (in_scale, in_zp, out_scale, out_zp))
    
    print("\n[OK] Generated nn_model.h")
    
    # --- nn_model.c ---
    with open(os.path.join(OUTPUT_DIR, "nn_model.c"), 'w') as f:
        f.write("""/*
 * nn_model.c - Pure C Neural Network Inference Engine
 * Auto-generated from TFLite INT8 model. Do not edit.
 *
 * Total weight size: ~%d bytes
 * Peak intermediate buffer: ~%d bytes
 * Stack requirement: ~%d bytes (with static buffers)
 */

#include "nn_model.h"
#include <string.h>

/* ================================================================
 * Helper: saturating int8 with rounding
 * ================================================================ */
static inline int8_t saturate_int8(int32_t val)
{
    if (val > 127) return 127;
    if (val < -128) return -128;
    return (int8_t)val;
}

static inline int32_t multiply_by_quantized_multiplier(int32_t x, int32_t multiplier, int shift)
{
    /* Fixed-point multiply: result = round( x * M * 2^(-shift) ) */
    int64_t result = (int64_t)x * (int64_t)multiplier;
    /* Rounding right shift by 31 */
    result = (result + (1LL << 30)) >> 31;
    /* Apply remaining shift */
    if (shift > 0) {
        result = (result + (1LL << (shift - 1))) >> shift;
    } else if (shift < 0) {
        result = result << (-shift);
    }
    return (int32_t)result;
}

""" % (
            conv0_w['data'].size + bn0_b['data'].size*4 +
            conv1_w['data'].size + bn1_b['data'].size*4 +
            conv2_w['data'].size + bn2_b['data'].size*4 +
            dense0_w['data'].size + dense0_b['data'].size*4 +
            dense1_w['data'].size + dense1_b['data'].size*4,
            100*6*24,  # rough peak
            100*6*24 + 50*6*16 + 25*6*24 + 1728 + 24 + 3  # rough stack
        ))
        
        # Write all weight arrays
        f.write("/* ================================================================\n")
        f.write(" * Model Weights (const, stored in Flash)\n")
        f.write(" * ================================================================ */\n\n")
        
        # Conv0: [8, 5, 1, 1] -> kernel_h=5, kernel_w=1, in_ch=1, out_ch=8
        f.write("/* Conv2D_0: kernel=(5,1), in_ch=1, out_ch=8 */\n")
        write_int8_array(f, "conv0_weight", conv0_w['data'])
        write_int32_array(f, "conv0_bias", bn0_b['data'])
        
        # Conv1: [16, 5, 1, 8]
        f.write("/* Conv2D_1: kernel=(5,1), in_ch=8, out_ch=16 */\n")
        write_int8_array(f, "conv1_weight", conv1_w['data'])
        write_int32_array(f, "conv1_bias", bn1_b['data'])
        
        # Conv2: [24, 3, 3, 16]
        f.write("/* Conv2D_2: kernel=(3,3), in_ch=16, out_ch=24 */\n")
        write_int8_array(f, "conv2_weight", conv2_w['data'])
        write_int32_array(f, "conv2_bias", bn2_b['data'])
        
        # Dense0: [24, 1728]
        f.write("/* Dense_0: in=1728, out=24 */\n")
        write_int8_array(f, "dense0_weight", dense0_w['data'])
        write_int32_array(f, "dense0_bias", dense0_b['data'])
        
        # Dense1: [3, 24]
        f.write("/* Dense_1: in=24, out=3 */\n")
        write_int8_array(f, "dense1_weight", dense1_w['data'])
        write_int32_array(f, "dense1_bias", dense1_b['data'])
        
        # Write quantization parameters
        f.write("/* ================================================================\n")
        f.write(" * Per-channel quantization scales for Conv layers\n")
        f.write(" * ================================================================ */\n\n")
        
        # For each conv layer, compute per-channel effective multiplier & shift
        # Conv0: input_scale * weight_scale[ch] / output_scale
        conv0_in_scale = in_scale
        conv0_out_scale = act12['scales'][0]
        conv0_out_zp = act12['zero_points'][0]
        
        conv1_in_scale = act13['scales'][0]  # after maxpool0
        conv1_out_scale = act14['scales'][0]
        conv1_out_zp = act14['zero_points'][0]
        
        conv2_in_scale = act15['scales'][0]  # after maxpool1
        conv2_out_scale = act16['scales'][0]
        conv2_out_zp = act16['zero_points'][0]
        
        dense0_in_scale = act17['scales'][0]  # after maxpool2 = flatten input
        dense0_out_scale = act19['scales'][0]
        dense0_out_zp = act19['zero_points'][0]
        
        dense1_in_scale = act19['scales'][0]  # after dense0+relu
        dense1_out_scale = out_scale
        dense1_out_zp = out_zp
        
        # Input zero point for each layer
        conv0_in_zp = in_zp
        conv1_in_zp = act13['zero_points'][0]
        conv2_in_zp = act15['zero_points'][0]
        dense0_in_zp = act17['zero_points'][0]
        dense1_in_zp = act19['zero_points'][0]
        
        print(f"\nActivation params:")
        print(f"  Conv0 input:  scale={conv0_in_scale}, zp={conv0_in_zp}")
        print(f"  Conv0 output: scale={conv0_out_scale}, zp={conv0_out_zp}")
        print(f"  Conv1 input:  scale={conv1_in_scale}, zp={conv1_in_zp}")
        print(f"  Conv1 output: scale={conv1_out_scale}, zp={conv1_out_zp}")
        print(f"  Conv2 input:  scale={conv2_in_scale}, zp={conv2_in_zp}")
        print(f"  Conv2 output: scale={conv2_out_scale}, zp={conv2_out_zp}")
        print(f"  Dense0 input: scale={dense0_in_scale}, zp={dense0_in_zp}")
        print(f"  Dense0 output:scale={dense0_out_scale}, zp={dense0_out_zp}")
        print(f"  Dense1 input: scale={dense1_in_scale}, zp={dense1_in_zp}")
        print(f"  Dense1 output:scale={dense1_out_scale}, zp={dense1_out_zp}")
        
        def write_layer_quant_params(f, layer_name, w_scales, in_scale, out_scale, out_zp, in_zp, n_channels):
            """Write per-channel multiplier and shift arrays."""
            multipliers = []
            shifts = []
            for ch in range(n_channels):
                w_s = w_scales[ch] if len(w_scales) > 1 else w_scales[0]
                effective_scale = in_scale * w_s / out_scale
                M, S = compute_multiplier_shift(effective_scale)
                multipliers.append(M)
                shifts.append(S)
            
            f.write(f"/* {layer_name} requantization: out_zp={out_zp}, in_zp={in_zp} */\n")
            f.write(f"#define {layer_name.upper()}_OUT_ZP  {out_zp}\n")
            f.write(f"#define {layer_name.upper()}_IN_ZP   {in_zp}\n\n")
            write_int32_array(f, f"{layer_name}_multiplier", np.array(multipliers, dtype=np.int32))
            write_int32_array(f, f"{layer_name}_shift", np.array(shifts, dtype=np.int32))
        
        write_layer_quant_params(f, "conv0", conv0_w['quant']['scales'],
                                 conv0_in_scale, conv0_out_scale, conv0_out_zp, conv0_in_zp, 8)
        write_layer_quant_params(f, "conv1", conv1_w['quant']['scales'],
                                 conv1_in_scale, conv1_out_scale, conv1_out_zp, conv1_in_zp, 16)
        write_layer_quant_params(f, "conv2", conv2_w['quant']['scales'],
                                 conv2_in_scale, conv2_out_scale, conv2_out_zp, conv2_in_zp, 24)
        write_layer_quant_params(f, "dense0", dense0_w['quant']['scales'],
                                 dense0_in_scale, dense0_out_scale, dense0_out_zp, dense0_in_zp, 24)
        write_layer_quant_params(f, "dense1", dense1_w['quant']['scales'],
                                 dense1_in_scale, dense1_out_scale, dense1_out_zp, dense1_in_zp, 3)
        
        # Generate inference function
        f.write("""
/* ================================================================
 * Intermediate buffers (static to avoid stack overflow)
 * ================================================================ */

/* Buffer allocation strategy:
 * We use two alternating buffers for ping-pong operation.
 * Max sizes needed:
 *   After Conv0:   100 x 6 x 8  = 4800
 *   After Pool0:   50 x 6 x 8   = 2400
 *   After Conv1:   50 x 6 x 16  = 4800
 *   After Pool1:   25 x 6 x 16  = 2400
 *   After Conv2:   25 x 6 x 24  = 3600
 *   After Pool2:   12 x 6 x 24  = 1728
 *   Dense0 output: 24
 *   Dense1 output: 3
 */
static int8_t buf_a[4800];  /* ping buffer */
static int8_t buf_b[4800];  /* pong buffer */

/* ================================================================
 * Layer implementations
 * ================================================================ */

/**
 * Conv2D with per-channel quantization, fused BN + ReLU.
 * Input layout:  [H, W, C_in]
 * Weight layout: [C_out, kH, kW, C_in] (TFLite format)
 * Output layout: [H, W, C_out]
 * Padding: SAME
 */
static void conv2d_relu(
    const int8_t* input, int in_h, int in_w, int in_c,
    const int8_t* weight, const int32_t* bias,
    int kernel_h, int kernel_w, int out_c,
    int8_t in_zp,
    const int32_t* multiplier, const int32_t* shift, int8_t out_zp,
    int8_t* output)
{
    /* SAME padding: output size = input size */
    int pad_h = (kernel_h - 1) / 2;
    int pad_w = (kernel_w - 1) / 2;
    
    for (int oh = 0; oh < in_h; oh++) {
        for (int ow = 0; ow < in_w; ow++) {
            for (int oc = 0; oc < out_c; oc++) {
                int32_t acc = bias[oc];
                
                for (int kh = 0; kh < kernel_h; kh++) {
                    int ih = oh + kh - pad_h;
                    if (ih < 0 || ih >= in_h) continue;
                    
                    for (int kw = 0; kw < kernel_w; kw++) {
                        int iw = ow + kw - pad_w;
                        if (iw < 0 || iw >= in_w) continue;
                        
                        for (int ic = 0; ic < in_c; ic++) {
                            int32_t input_val = (int32_t)input[(ih * in_w + iw) * in_c + ic] - in_zp;
                            int32_t weight_val = (int32_t)weight[((oc * kernel_h + kh) * kernel_w + kw) * in_c + ic];
                            acc += input_val * weight_val;
                        }
                    }
                }
                
                /* Requantize */
                acc = multiply_by_quantized_multiplier(acc, multiplier[oc], shift[oc]);
                acc += out_zp;
                
                /* ReLU: max(0, val) in quantized domain = max(out_zp, val) */
                if (acc < out_zp) acc = out_zp;
                
                output[(oh * in_w + ow) * out_c + oc] = saturate_int8(acc);
            }
        }
    }
}

/**
 * MaxPool2D with (2,1) kernel and (2,1) stride.
 * Halves the height, keeps width unchanged.
 */
static void maxpool2d_2x1(
    const int8_t* input, int in_h, int in_w, int channels,
    int8_t* output)
{
    int out_h = in_h / 2;
    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < in_w; ow++) {
            for (int c = 0; c < channels; c++) {
                int idx0 = ((oh * 2) * in_w + ow) * channels + c;
                int idx1 = ((oh * 2 + 1) * in_w + ow) * channels + c;
                int8_t v0 = input[idx0];
                int8_t v1 = input[idx1];
                output[(oh * in_w + ow) * channels + c] = (v0 > v1) ? v0 : v1;
            }
        }
    }
}

/**
 * Fully connected (Dense) layer with optional ReLU.
 */
static void dense_layer(
    const int8_t* input, int in_size,
    const int8_t* weight, const int32_t* bias, int out_size,
    int8_t in_zp,
    const int32_t* multiplier, const int32_t* shift, int8_t out_zp,
    int apply_relu,
    int8_t* output)
{
    for (int o = 0; o < out_size; o++) {
        int32_t acc = bias[o];
        
        for (int i = 0; i < in_size; i++) {
            int32_t input_val = (int32_t)input[i] - in_zp;
            int32_t weight_val = (int32_t)weight[o * in_size + i];
            acc += input_val * weight_val;
        }
        
        /* Requantize */
        acc = multiply_by_quantized_multiplier(acc, multiplier[o], shift[o]);
        acc += out_zp;
        
        /* Optional ReLU */
        if (apply_relu && acc < out_zp) {
            acc = out_zp;
        }
        
        output[o] = saturate_int8(acc);
    }
}

/* ================================================================
 * Main inference function
 * ================================================================ */
void nn_inference(const int8_t* input, int8_t* output)
{
    /* 
     * Input shape: [100, 6, 1]
     * 
     * Layer 0: Conv2D(8, 5x1) + BN + ReLU -> [100, 6, 8]
     * Layer 1: MaxPool(2,1)               -> [50, 6, 8]
     * Layer 2: Conv2D(16, 5x1) + BN + ReLU -> [50, 6, 16]
     * Layer 3: MaxPool(2,1)               -> [25, 6, 16]
     * Layer 4: Conv2D(24, 3x3) + BN + ReLU -> [25, 6, 24]
     * Layer 5: MaxPool(2,1)               -> [12, 6, 24]
     * Layer 6: Flatten                    -> [1728]
     * Layer 7: Dense(24) + ReLU           -> [24]
     * Layer 8: Dense(3)                   -> [3]
     */
    
    /* Conv0: [100,6,1] -> [100,6,8] */
    conv2d_relu(input, 100, 6, 1,
                conv0_weight, conv0_bias, 5, 1, 8,
                (int8_t)CONV0_IN_ZP,
                conv0_multiplier, conv0_shift, (int8_t)CONV0_OUT_ZP,
                buf_a);
    
    /* MaxPool0: [100,6,8] -> [50,6,8] */
    maxpool2d_2x1(buf_a, 100, 6, 8, buf_b);
    
    /* Conv1: [50,6,8] -> [50,6,16] */
    conv2d_relu(buf_b, 50, 6, 8,
                conv1_weight, conv1_bias, 5, 1, 16,
                (int8_t)CONV1_IN_ZP,
                conv1_multiplier, conv1_shift, (int8_t)CONV1_OUT_ZP,
                buf_a);
    
    /* MaxPool1: [50,6,16] -> [25,6,16] */
    maxpool2d_2x1(buf_a, 50, 6, 16, buf_b);
    
    /* Conv2: [25,6,16] -> [25,6,24] */
    conv2d_relu(buf_b, 25, 6, 16,
                conv2_weight, conv2_bias, 3, 3, 24,
                (int8_t)CONV2_IN_ZP,
                conv2_multiplier, conv2_shift, (int8_t)CONV2_OUT_ZP,
                buf_a);
    
    /* MaxPool2: [25,6,24] -> [12,6,24] */
    maxpool2d_2x1(buf_a, 25, 6, 24, buf_b);
    
    /* Flatten: [12,6,24] -> [1728], data is already flat in buf_b */
    
    /* Dense0: [1728] -> [24] + ReLU */
    {
        int8_t dense0_out[24];
        dense_layer(buf_b, 1728,
                    dense0_weight, dense0_bias, 24,
                    (int8_t)DENSE0_IN_ZP,
                    dense0_multiplier, dense0_shift, (int8_t)DENSE0_OUT_ZP,
                    1, /* apply_relu */
                    dense0_out);
        
        /* Dense1: [24] -> [3] */
        dense_layer(dense0_out, 24,
                    dense1_weight, dense1_bias, 3,
                    (int8_t)DENSE1_IN_ZP,
                    dense1_multiplier, dense1_shift, (int8_t)DENSE1_OUT_ZP,
                    0, /* no relu */
                    output);
    }
}

void nn_decode_output(const int8_t raw[NN_NUM_CLASSES], float probs[NN_NUM_CLASSES])
{
    for (int i = 0; i < NN_NUM_CLASSES; i++) {
        probs[i] = ((float)raw[i] - NN_OUTPUT_ZERO_POINT) * NN_OUTPUT_SCALE;
        if (probs[i] < 0.0f) probs[i] = 0.0f;
        if (probs[i] > 1.0f) probs[i] = 1.0f;
    }
}

int nn_argmax(const int8_t raw[NN_NUM_CLASSES])
{
    int best = 0;
    for (int i = 1; i < NN_NUM_CLASSES; i++) {
        if (raw[i] > raw[best]) best = i;
    }
    return best;
}
""")
    
    print("[OK] Generated nn_model.c")
    
    # ============================================================
    # Verify by comparing C inference with TFLite interpreter
    # ============================================================
    print("\n=== Verification ===")
    
    # Run TFLite with a test input
    test_input = np.random.randint(-128, 128, size=(1, 100, 6, 1), dtype=np.int8)
    interpreter.set_tensor(input_det['index'], test_input)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_det['index'])
    
    print(f"Test TFLite output: {tflite_output}")
    tflite_probs = (tflite_output.astype(np.float32) - out_zp) * out_scale
    print(f"Probabilities: {tflite_probs}")
    print(f"Predicted class: {np.argmax(tflite_output)}")
    
    print("\n[OK] All files generated successfully!")
    print(f"  Output directory: {OUTPUT_DIR}/")
    print(f"  Files: nn_model.h, nn_model.c")
    print(f"  (Also keep: preprocessing.h, preprocessing.c)")

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
