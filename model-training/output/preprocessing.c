/*
 * Movement Type Detection - Preprocessing Implementation
 */

#include "preprocessing.h"
#include <string.h>

static float    s_window_buf[PREPROC_WINDOW_SIZE][PREPROC_NUM_CHANNELS];
static int8_t   s_model_input[PREPROC_BUFFER_LEN];
static int      s_write_idx;
static int      s_samples_since_inference;
static bool     s_buffer_full;

void preprocess_init(void)
{
    memset(s_window_buf, 0, sizeof(s_window_buf));
    memset(s_model_input, 0, sizeof(s_model_input));
    s_write_idx = 0;
    s_samples_since_inference = 0;
    s_buffer_full = false;
}

static float normalize_channel(float val, int ch)
{
    float range = CHANNEL_MAX[ch] - CHANNEL_MIN[ch];
    if (range < 1e-8f) return 0.0f;
    float norm = (val - CHANNEL_MIN[ch]) / range;
    if (norm < 0.0f) norm = 0.0f;
    if (norm > 1.0f) norm = 1.0f;
    return norm;
}

static int8_t float_to_int8(float val)
{
    int32_t quantized = (int32_t)(val / INPUT_SCALE) + INPUT_ZERO_POINT;
    if (quantized < -128) quantized = -128;
    if (quantized >  127) quantized = 127;
    return (int8_t)quantized;
}

bool preprocess_add_sample(const float sample[PREPROC_NUM_CHANNELS])
{
    for (int ch = 0; ch < PREPROC_NUM_CHANNELS; ch++) {
        s_window_buf[s_write_idx][ch] = sample[ch];
    }

    s_write_idx++;
    s_samples_since_inference++;

    if (s_write_idx >= PREPROC_WINDOW_SIZE) {
        s_buffer_full = true;
    }

    if (s_buffer_full && s_samples_since_inference >= PREPROC_WINDOW_STEP) {
        int read_start = s_write_idx - PREPROC_WINDOW_SIZE;
        if (read_start < 0) read_start = 0;

        int out_idx = 0;
        for (int t = read_start; t < read_start + PREPROC_WINDOW_SIZE; t++) {
            for (int ch = 0; ch < PREPROC_NUM_CHANNELS; ch++) {
                float norm_val = normalize_channel(s_window_buf[t][ch], ch);
                s_model_input[out_idx++] = float_to_int8(norm_val);
            }
        }

        s_samples_since_inference = 0;

        if (s_write_idx >= PREPROC_WINDOW_SIZE) {
            int keep_from = PREPROC_WINDOW_STEP;
            memmove(s_window_buf[0], s_window_buf[keep_from],
                    (PREPROC_WINDOW_SIZE - PREPROC_WINDOW_STEP) * PREPROC_NUM_CHANNELS * sizeof(float));
            s_write_idx = PREPROC_WINDOW_SIZE - PREPROC_WINDOW_STEP;
        }

        return true;
    }

    return false;
}

const int8_t* preprocess_get_model_input(void)
{
    return s_model_input;
}

void preprocess_decode_output(const int8_t raw_output[MODEL_NUM_CLASSES],
                              float probs[MODEL_NUM_CLASSES])
{
    for (int i = 0; i < MODEL_NUM_CLASSES; i++) {
        probs[i] = ((float)raw_output[i] - OUTPUT_ZERO_POINT) * OUTPUT_SCALE;
        if (probs[i] < 0.0f) probs[i] = 0.0f;
        if (probs[i] > 1.0f) probs[i] = 1.0f;
    }
}

int preprocess_argmax(const float probs[MODEL_NUM_CLASSES])
{
    int best_idx = 0;
    float best_val = probs[0];
    for (int i = 1; i < MODEL_NUM_CLASSES; i++) {
        if (probs[i] > best_val) {
            best_val = probs[i];
            best_idx = i;
        }
    }
    return best_idx;
}
