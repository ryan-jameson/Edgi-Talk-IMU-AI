"""
Movement Type Detection - Full Training Pipeline Script
========================================================
Runs the complete training pipeline: data loading, windowing, training,
quantization, and C code generation.

Usage:
    python run_training.py
"""

import os, sys, glob, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============ Configuration ============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, 'Data')
SAMPLE_RATE = 50
WINDOW_SEC = 2.0
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_SEC)  # 100
STEP_SIZE = WINDOW_SIZE // 2                  # 50
NUM_CHANNELS = 6
LABEL_MAP = {'stationary': 0, 'shaking': 1, 'circle': 2}
LABEL_NAMES = ['stationary', 'shaking', 'circle']
NUM_CLASSES = len(LABEL_NAMES)
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 5e-4
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print('='*65)
print('  Movement Type Detection — Training Pipeline')
print('='*65)
print(f'  Window: {WINDOW_SIZE} samples ({WINDOW_SEC}s), Step: {STEP_SIZE}')
print(f'  Classes: {LABEL_NAMES}')
print()

# ============ Data Loading ============
print('[1/10] Loading data...')

def load_imu_data(filepath):
    df = pd.read_csv(filepath, comment='#',
                     names=['time', 'accel_x', 'accel_y', 'accel_z',
                            'gyro_x', 'gyro_y', 'gyro_z'])
    return df

def load_labels(filepath):
    df = pd.read_csv(filepath)
    labels = []
    for _, row in df.iterrows():
        start = row.iloc[0]
        length = row.iloc[1]
        label = str(row.iloc[2]).strip().lower()
        end = start + length
        labels.append((start, end, label))
    return labels

def assign_labels_to_samples(imu_df, label_intervals):
    times = imu_df['time'].values
    labels = np.full(len(times), 'stationary', dtype=object)
    for start, end, label in label_intervals:
        mask = (times >= start) & (times < end)
        labels[mask] = label
    return labels

def discover_sessions(data_root):
    sessions = []
    for person_dir in sorted(glob.glob(os.path.join(data_root, '*'))):
        if not os.path.isdir(person_dir):
            continue
        person_name = os.path.basename(person_dir)
        for session_dir in sorted(glob.glob(os.path.join(person_dir, '*'))):
            if not os.path.isdir(session_dir):
                continue
            imu_file = os.path.join(session_dir, 'IMU-Data.data')
            label_file = os.path.join(session_dir, 'Live-Labeling.label')
            if os.path.isfile(imu_file) and os.path.isfile(label_file):
                sessions.append({
                    'person': person_name,
                    'session': os.path.basename(session_dir),
                    'imu_path': imu_file,
                    'label_path': label_file
                })
    return sessions

sessions = discover_sessions(DATA_ROOT)
print(f'  Found {len(sessions)} sessions')

all_segments = []
for s in sessions:
    imu_df = load_imu_data(s['imu_path'])
    label_intervals = load_labels(s['label_path'])
    sample_labels = assign_labels_to_samples(imu_df, label_intervals)
    imu_values = imu_df[['accel_x', 'accel_y', 'accel_z',
                         'gyro_x', 'gyro_y', 'gyro_z']].values
    all_segments.append({
        'person': s['person'],
        'session': s['session'],
        'imu': imu_values,
        'labels': sample_labels,
        'n_samples': len(imu_values)
    })

total_samples = sum(seg['n_samples'] for seg in all_segments)
print(f'  Total samples: {total_samples} ({total_samples/SAMPLE_RATE:.1f}s)')

label_counts = {lbl: 0 for lbl in LABEL_NAMES}
for seg in all_segments:
    for lbl in LABEL_NAMES:
        label_counts[lbl] += int(np.sum(seg['labels'] == lbl))
for lbl, cnt in label_counts.items():
    print(f'    {lbl:12s}: {cnt:>8d} samples ({cnt/SAMPLE_RATE:.1f}s)')

# ============ Windowing ============
print('\n[2/10] Creating sliding windows...')

def create_windows(imu_data, labels, window_size, step_size, label_threshold=0.7):
    X_windows = []
    y_labels = []
    n_samples = len(imu_data)
    for start in range(0, n_samples - window_size + 1, step_size):
        end = start + window_size
        window_data = imu_data[start:end]
        window_labels = labels[start:end]
        unique, counts = np.unique(window_labels, return_counts=True)
        proportions = dict(zip(unique, counts / window_size))
        dominant_label = max(proportions, key=proportions.get)
        dominant_ratio = proportions[dominant_label]
        if dominant_ratio >= label_threshold:
            X_windows.append(window_data)
            y_labels.append(LABEL_MAP[dominant_label])
    return np.array(X_windows) if X_windows else np.empty((0, window_size, 6)), np.array(y_labels)

X_all = []
y_all = []
for seg in all_segments:
    X_win, y_win = create_windows(seg['imu'], seg['labels'], WINDOW_SIZE, STEP_SIZE)
    if len(X_win) > 0:
        X_all.append(X_win)
        y_all.append(y_win)

X_all = np.concatenate(X_all, axis=0)
y_all = np.concatenate(y_all, axis=0)

print(f'  Total windows: {len(X_all)}')
print(f'  X shape: {X_all.shape}')
for i, name in enumerate(LABEL_NAMES):
    print(f'    {name:12s}: {np.sum(y_all == i):>6d} windows')

# ============ Normalization ============
print('\n[3/10] Normalizing data...')

ch_names = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']
channel_min = X_all.reshape(-1, NUM_CHANNELS).min(axis=0)
channel_max = X_all.reshape(-1, NUM_CHANNELS).max(axis=0)

for i, name in enumerate(ch_names):
    print(f'    {name:8s}: min={channel_min[i]:>10.4f}  max={channel_max[i]:>10.4f}')

X_normalized = (X_all - channel_min) / (channel_max - channel_min + 1e-8)
norm_params = {
    'channel_names': ch_names,
    'min': channel_min.tolist(),
    'max': channel_max.tolist()
}
print(f'  Normalized range: [{X_normalized.min():.4f}, {X_normalized.max():.4f}]')

# ============ Data Split ============
print('\n[4/10] Splitting data (80/10/10)...')
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

X_train, X_temp, y_train, y_temp = train_test_split(
    X_normalized, y_all, test_size=0.2, random_state=42, stratify=y_all)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f'  Train: {X_train.shape[0]:>6d} ({X_train.shape[0]/len(X_normalized)*100:.1f}%)')
print(f'  Val:   {X_val.shape[0]:>6d} ({X_val.shape[0]/len(X_normalized)*100:.1f}%)')
print(f'  Test:  {X_test.shape[0]:>6d} ({X_test.shape[0]/len(X_normalized)*100:.1f}%)')

class_weights_array = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=y_train)
class_weights = dict(enumerate(class_weights_array))
print('  Class weights:', {LABEL_NAMES[i]: f'{w:.3f}' for i, w in class_weights.items()})

# ============ Build Model ============
print('\n[5/10] Building model...')
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.keras import layers, models, regularizers

X_train_4d = X_train[..., np.newaxis]
X_val_4d = X_val[..., np.newaxis]
X_test_4d = X_test[..., np.newaxis]

def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(8, (5, 1), padding='same', use_bias=False,
                      input_shape=input_shape),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 1)),
        layers.Conv2D(16, (5, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 1)),
        layers.Conv2D(24, (3, 3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 1)),
        layers.Flatten(),
        layers.Dense(24, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = build_model((WINDOW_SIZE, NUM_CHANNELS, 1), NUM_CLASSES)
model.summary()

# ============ Train ============
print('\n[6/10] Training model...')
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=25, restore_best_weights=True, verbose=1,
        mode='max'),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(OUTPUT_DIR, 'best_model.keras'),
        monitor='val_accuracy', save_best_only=True, verbose=1, mode='max')
]

history = model.fit(
    X_train_4d, y_train,
    validation_data=(X_val_4d, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=2
)

# ============ Evaluate ============
print('\n[7/10] Evaluating model...')
from sklearn.metrics import classification_report, confusion_matrix

test_loss, test_acc = model.evaluate(X_test_4d, y_test, verbose=0)
print(f'  Test Loss:     {test_loss:.4f}')
print(f'  Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)')

y_pred = model.predict(X_test_4d, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
print('\n' + classification_report(y_test, y_pred_classes, target_names=LABEL_NAMES))

cm = confusion_matrix(y_test, y_pred_classes)
print('Confusion Matrix:')
print(cm)

# ============ TFLite Conversion ============
print('\n[8/10] Converting to TFLite (INT8 quantization)...')

def representative_dataset_gen():
    indices = np.random.choice(len(X_train_4d), size=min(200, len(X_train_4d)), replace=False)
    for i in indices:
        sample = X_train_4d[i:i+1].astype(np.float32)
        yield [sample]

# Float32
converter_f32 = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_f32 = converter_f32.convert()
f32_path = os.path.join(OUTPUT_DIR, 'model_float32.tflite')
with open(f32_path, 'wb') as f:
    f.write(tflite_f32)

# INT8
converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_dataset_gen
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type = tf.int8
converter_int8.inference_output_type = tf.int8
tflite_int8 = converter_int8.convert()
int8_path = os.path.join(OUTPUT_DIR, 'model_int8.tflite')
with open(int8_path, 'wb') as f:
    f.write(tflite_int8)

print(f'  Float32 size: {os.path.getsize(f32_path)/1024:.1f} KB')
print(f'  INT8 size:    {os.path.getsize(int8_path)/1024:.1f} KB')

# Validate INT8
interpreter = tf.lite.Interpreter(model_path=int8_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_scale = input_details[0]['quantization'][0]
input_zero_point = input_details[0]['quantization'][1]
output_scale = output_details[0]['quantization'][0]
output_zero_point = output_details[0]['quantization'][1]

print(f'  Input quant:  scale={input_scale}, zero_point={input_zero_point}')
print(f'  Output quant: scale={output_scale}, zero_point={output_zero_point}')

correct = 0
for i in range(len(X_test_4d)):
    input_data = X_test_4d[i:i+1].astype(np.float32)
    input_quantized = (input_data / input_scale + input_zero_point).astype(np.int8)
    interpreter.set_tensor(input_details[0]['index'], input_quantized)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    if np.argmax(output_data) == y_test[i]:
        correct += 1

int8_accuracy = correct / len(X_test_4d)
print(f'  INT8 Test Accuracy: {int8_accuracy:.4f} ({int8_accuracy*100:.1f}%)')
print(f'  Accuracy drop:      {(test_acc - int8_accuracy)*100:.2f}%')

# ============ Generate C Files ============
print('\n[9/10] Generating C files...')

with open(int8_path, 'rb') as f:
    model_data = f.read()
model_size = len(model_data)

c_array_lines = []
for i in range(0, model_size, 12):
    chunk = model_data[i:i+12]
    hex_str = ', '.join(f'0x{b:02x}' for b in chunk)
    c_array_lines.append(f'  {hex_str},')
if c_array_lines:
    c_array_lines[-1] = c_array_lines[-1].rstrip(',')

# model.h
model_h = f"""/*
 * Movement Type Detection - TFLite Micro Model
 *
 * Auto-generated. Do not edit.
 * Input:  {WINDOW_SIZE} timesteps x {NUM_CHANNELS} channels (100x6x1), INT8
 * Output: {NUM_CLASSES} classes [{', '.join(LABEL_NAMES)}]
 * Size:   {model_size} bytes ({model_size/1024:.1f} KB)
 *
 * Sampling Rate:  {SAMPLE_RATE} Hz
 * Window Size:    {WINDOW_SIZE} samples ({WINDOW_SEC}s)
 * Window Step:    {STEP_SIZE} samples ({STEP_SIZE/SAMPLE_RATE}s)
 */

#ifndef MODEL_H_
#define MODEL_H_

#ifdef __cplusplus
extern "C" {{
#endif

extern const unsigned char g_model[];
extern const unsigned int  g_model_len;

#define MODEL_INPUT_SAMPLES   {WINDOW_SIZE}
#define MODEL_INPUT_CHANNELS  {NUM_CHANNELS}
#define MODEL_NUM_CLASSES     {NUM_CLASSES}
#define MODEL_SAMPLE_RATE     {SAMPLE_RATE}
#define MODEL_WINDOW_STEP     {STEP_SIZE}

#define CLASS_STATIONARY  0
#define CLASS_SHAKING     1
#define CLASS_CIRCLE      2

static const char* const g_class_names[MODEL_NUM_CLASSES] = {{
    "stationary",
    "shaking",
    "circle"
}};

#define INPUT_SCALE       {input_scale}f
#define INPUT_ZERO_POINT  {input_zero_point}
#define OUTPUT_SCALE       {output_scale}f
#define OUTPUT_ZERO_POINT  {output_zero_point}

#ifdef __cplusplus
}}
#endif

#endif  /* MODEL_H_ */
"""

with open(os.path.join(OUTPUT_DIR, 'model.h'), 'w') as f:
    f.write(model_h)

# model.c
model_c = f"""/*
 * Movement Type Detection - TFLite Micro Model Data
 * Auto-generated. Do not edit.
 * Size: {model_size} bytes ({model_size/1024:.1f} KB)
 */

#include "model.h"

#if defined(__GNUC__)
__attribute__((aligned(16)))
#endif
const unsigned char g_model[] = {{
{chr(10).join(c_array_lines)}
}};

const unsigned int g_model_len = {model_size};
"""

with open(os.path.join(OUTPUT_DIR, 'model.c'), 'w') as f:
    f.write(model_c)

# preprocessing.h
min_str = ', '.join(f'{v:.6f}f' for v in channel_min)
max_str = ', '.join(f'{v:.6f}f' for v in channel_max)

preproc_h = f"""/*
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
#include "model.h"

#ifdef __cplusplus
extern "C" {{
#endif

#define PREPROC_WINDOW_SIZE    MODEL_INPUT_SAMPLES
#define PREPROC_NUM_CHANNELS   MODEL_INPUT_CHANNELS
#define PREPROC_WINDOW_STEP    MODEL_WINDOW_STEP
#define PREPROC_BUFFER_LEN     (PREPROC_WINDOW_SIZE * PREPROC_NUM_CHANNELS)

/* Channel order: Accel_X, Accel_Y, Accel_Z, Gyro_X, Gyro_Y, Gyro_Z */
static const float CHANNEL_MIN[PREPROC_NUM_CHANNELS] = {{ {min_str} }};
static const float CHANNEL_MAX[PREPROC_NUM_CHANNELS] = {{ {max_str} }};

void preprocess_init(void);
bool preprocess_add_sample(const float sample[PREPROC_NUM_CHANNELS]);
const int8_t* preprocess_get_model_input(void);
void preprocess_decode_output(const int8_t raw_output[MODEL_NUM_CLASSES],
                              float probs[MODEL_NUM_CLASSES]);
int preprocess_argmax(const float probs[MODEL_NUM_CLASSES]);

#ifdef __cplusplus
}}
#endif

#endif  /* PREPROCESSING_H_ */
"""

with open(os.path.join(OUTPUT_DIR, 'preprocessing.h'), 'w') as f:
    f.write(preproc_h)

# preprocessing.c
preproc_c = f"""/*
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
{{
    memset(s_window_buf, 0, sizeof(s_window_buf));
    memset(s_model_input, 0, sizeof(s_model_input));
    s_write_idx = 0;
    s_samples_since_inference = 0;
    s_buffer_full = false;
}}

static float normalize_channel(float val, int ch)
{{
    float range = CHANNEL_MAX[ch] - CHANNEL_MIN[ch];
    if (range < 1e-8f) return 0.0f;
    float norm = (val - CHANNEL_MIN[ch]) / range;
    if (norm < 0.0f) norm = 0.0f;
    if (norm > 1.0f) norm = 1.0f;
    return norm;
}}

static int8_t float_to_int8(float val)
{{
    int32_t quantized = (int32_t)(val / INPUT_SCALE) + INPUT_ZERO_POINT;
    if (quantized < -128) quantized = -128;
    if (quantized >  127) quantized = 127;
    return (int8_t)quantized;
}}

bool preprocess_add_sample(const float sample[PREPROC_NUM_CHANNELS])
{{
    for (int ch = 0; ch < PREPROC_NUM_CHANNELS; ch++) {{
        s_window_buf[s_write_idx][ch] = sample[ch];
    }}

    s_write_idx++;
    s_samples_since_inference++;

    if (s_write_idx >= PREPROC_WINDOW_SIZE) {{
        s_buffer_full = true;
    }}

    if (s_buffer_full && s_samples_since_inference >= PREPROC_WINDOW_STEP) {{
        int read_start = s_write_idx - PREPROC_WINDOW_SIZE;
        if (read_start < 0) read_start = 0;

        int out_idx = 0;
        for (int t = read_start; t < read_start + PREPROC_WINDOW_SIZE; t++) {{
            for (int ch = 0; ch < PREPROC_NUM_CHANNELS; ch++) {{
                float norm_val = normalize_channel(s_window_buf[t][ch], ch);
                s_model_input[out_idx++] = float_to_int8(norm_val);
            }}
        }}

        s_samples_since_inference = 0;

        if (s_write_idx >= PREPROC_WINDOW_SIZE) {{
            int keep_from = PREPROC_WINDOW_STEP;
            memmove(s_window_buf[0], s_window_buf[keep_from],
                    (PREPROC_WINDOW_SIZE - PREPROC_WINDOW_STEP) * PREPROC_NUM_CHANNELS * sizeof(float));
            s_write_idx = PREPROC_WINDOW_SIZE - PREPROC_WINDOW_STEP;
        }}

        return true;
    }}

    return false;
}}

const int8_t* preprocess_get_model_input(void)
{{
    return s_model_input;
}}

void preprocess_decode_output(const int8_t raw_output[MODEL_NUM_CLASSES],
                              float probs[MODEL_NUM_CLASSES])
{{
    for (int i = 0; i < MODEL_NUM_CLASSES; i++) {{
        probs[i] = ((float)raw_output[i] - OUTPUT_ZERO_POINT) * OUTPUT_SCALE;
        if (probs[i] < 0.0f) probs[i] = 0.0f;
        if (probs[i] > 1.0f) probs[i] = 1.0f;
    }}
}}

int preprocess_argmax(const float probs[MODEL_NUM_CLASSES])
{{
    int best_idx = 0;
    float best_val = probs[0];
    for (int i = 1; i < MODEL_NUM_CLASSES; i++) {{
        if (probs[i] > best_val) {{
            best_val = probs[i];
            best_idx = i;
        }}
    }}
    return best_idx;
}}
"""

with open(os.path.join(OUTPUT_DIR, 'preprocessing.c'), 'w') as f:
    f.write(preproc_c)

print('  Generated: model.h')
print('  Generated: model.c')
print('  Generated: preprocessing.h')
print('  Generated: preprocessing.c')

# ============ Save Reports ============
print('\n[10/10] Saving reports...')

norm_params_path = os.path.join(OUTPUT_DIR, 'normalization_params.json')
with open(norm_params_path, 'w') as f:
    json.dump(norm_params, f, indent=2)

report = {
    'model_architecture': 'Conv2D (3 blocks) + GlobalAvgPool + Dense',
    'input_shape': [WINDOW_SIZE, NUM_CHANNELS, 1],
    'output_classes': LABEL_NAMES,
    'sample_rate_hz': SAMPLE_RATE,
    'window_size_samples': WINDOW_SIZE,
    'window_size_seconds': WINDOW_SEC,
    'window_step_samples': STEP_SIZE,
    'total_windows': int(len(X_all)),
    'train_windows': int(len(X_train)),
    'val_windows': int(len(X_val)),
    'test_windows': int(len(X_test)),
    'float32_test_accuracy': float(test_acc),
    'int8_test_accuracy': float(int8_accuracy),
    'float32_model_size_kb': round(os.path.getsize(f32_path) / 1024, 1),
    'int8_model_size_kb': round(os.path.getsize(int8_path) / 1024, 1),
    'input_quantization': {
        'scale': float(input_scale),
        'zero_point': int(input_zero_point)
    },
    'output_quantization': {
        'scale': float(output_scale),
        'zero_point': int(output_zero_point)
    },
    'normalization': norm_params
}

with open(os.path.join(OUTPUT_DIR, 'training_report.json'), 'w') as f:
    json.dump(report, f, indent=2)

# Final summary
print()
print('='*65)
print('       TRAINING COMPLETE — OUTPUT SUMMARY')
print('='*65)
print()
output_files = [
    'model.h', 'model.c', 'preprocessing.h', 'preprocessing.c',
    'model_int8.tflite', 'model_float32.tflite', 'best_model.keras',
    'normalization_params.json', 'training_report.json'
]
for fname in output_files:
    fpath = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(fpath):
        size_kb = os.path.getsize(fpath) / 1024
        print(f'  OK  {fname:<35s} {size_kb:>8.1f} KB')
    else:
        print(f'  --  {fname:<35s} (missing)')

print()
print(f'  Float32 Test Accuracy: {test_acc*100:.1f}%')
print(f'  INT8 Test Accuracy:    {int8_accuracy*100:.1f}%')
print(f'  INT8 Model Size:       {os.path.getsize(int8_path)/1024:.1f} KB')
print()
print('  Deployment files: model.h, model.c, preprocessing.h, preprocessing.c')
print('='*65)
