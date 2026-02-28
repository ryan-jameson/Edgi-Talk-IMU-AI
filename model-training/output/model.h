/*
 * Movement Type Detection - TFLite Micro Model
 *
 * Auto-generated. Do not edit.
 * Input:  100 timesteps x 6 channels (100x6x1), INT8
 * Output: 3 classes [stationary, shaking, circle]
 * Size:   51840 bytes (50.6 KB)
 *
 * Sampling Rate:  50 Hz
 * Window Size:    100 samples (2.0s)
 * Window Step:    50 samples (1.0s)
 */

#ifndef MODEL_H_
#define MODEL_H_

#ifdef __cplusplus
extern "C" {
#endif

extern const unsigned char g_model[];
extern const unsigned int  g_model_len;

#define MODEL_INPUT_SAMPLES   100
#define MODEL_INPUT_CHANNELS  6
#define MODEL_NUM_CLASSES     3
#define MODEL_SAMPLE_RATE     50
#define MODEL_WINDOW_STEP     50

#define CLASS_STATIONARY  0
#define CLASS_SHAKING     1
#define CLASS_CIRCLE      2

static const char* const g_class_names[MODEL_NUM_CLASSES] = {
    "stationary",
    "shaking",
    "circle"
};

#define INPUT_SCALE       0.003921568859368563f
#define INPUT_ZERO_POINT  -128
#define OUTPUT_SCALE       0.00390625f
#define OUTPUT_ZERO_POINT  -128

#ifdef __cplusplus
}
#endif

#endif  /* MODEL_H_ */
