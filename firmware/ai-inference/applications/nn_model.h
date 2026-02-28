/*
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
#define NN_INPUT_SCALE       3.9215688594e-03f
#define NN_INPUT_ZERO_POINT  -128

/* Output quantization: float_val = (q - zero_point) * scale */
#define NN_OUTPUT_SCALE       3.9062500000e-03f
#define NN_OUTPUT_ZERO_POINT  -128

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
