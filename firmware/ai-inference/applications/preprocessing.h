/*
 * Movement Type Detection - Preprocessing for Embedded Deployment
 *
 * Handles:
 *   1. Sliding window buffer management
 *   2. Min-Max normalization
 *   3. Float -> INT8 quantization for model input
 *
 * Usage:
 *   1. Call preprocess_init() once at startup
 *   2. For each new IMU sample, call preprocess_add_sample()
 *   3. When returns true, call preprocess_get_model_input() for inference
 */

#ifndef PREPROCESSING_H_
#define PREPROCESSING_H_

#include <stdint.h>
#include <stdbool.h>
#include "nn_model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PREPROC_WINDOW_SIZE    NN_INPUT_HEIGHT
#define PREPROC_NUM_CHANNELS   6    /* Accel_X/Y/Z + Gyro_X/Y/Z */
#define PREPROC_WINDOW_STEP    NN_WINDOW_STEP
#define PREPROC_BUFFER_LEN     (PREPROC_WINDOW_SIZE * PREPROC_NUM_CHANNELS)

/* Channel order: Accel_X, Accel_Y, Accel_Z, Gyro_X, Gyro_Y, Gyro_Z */
static const float CHANNEL_MIN[PREPROC_NUM_CHANNELS] = { -57.843914f, -52.114590f, -49.334919f, -500.000000f, -500.000000f, -500.000000f };
static const float CHANNEL_MAX[PREPROC_NUM_CHANNELS] = { 46.809036f, 51.446606f, 55.464077f, 499.984741f, 499.984741f, 499.984741f };

void preprocess_init(void);
bool preprocess_add_sample(const float sample[PREPROC_NUM_CHANNELS]);
const int8_t* preprocess_get_model_input(void);
void preprocess_decode_output(const int8_t raw_output[NN_NUM_CLASSES],
                              float probs[NN_NUM_CLASSES]);
int preprocess_argmax(const float probs[NN_NUM_CLASSES]);

#ifdef __cplusplus
}
#endif

#endif  /* PREPROCESSING_H_ */
