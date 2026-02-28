/* ai_task.c - AI inference (pure C INT8 CNN, 50Hz) */
#include <rtthread.h>
#include <rtdevice.h>
#include "nn_model.h"
#include "preprocessing.h"
#include "ui_record.h"

extern volatile int16_t g_acc_x, g_acc_y, g_acc_z;
extern volatile int16_t g_gyro_x, g_gyro_y, g_gyro_z;
extern void update_ai_label(const char *text, int user_id);

#define ACC_SENS   0.000598f
#define GYRO_SENS  0.0175f

static void ai_thread_entry(void *parameter)
{
    float sample[6];
    int8_t nn_out[NN_NUM_CLASSES];
    float probs[NN_NUM_CLASSES];
    int last_cls = -1;
    int stable = 0;
    int cls;

    (void)parameter;
    rt_kprintf("[AI] Engine started\n");
    preprocess_init();

    while (1) {
        sample[0] = (float)g_acc_x  * ACC_SENS;
        sample[1] = (float)g_acc_y  * ACC_SENS;
        sample[2] = (float)g_acc_z  * ACC_SENS;
        sample[3] = (float)g_gyro_x * GYRO_SENS;
        sample[4] = (float)g_gyro_y * GYRO_SENS;
        sample[5] = (float)g_gyro_z * GYRO_SENS;

        if (preprocess_add_sample(sample)) {
            const int8_t *in = preprocess_get_model_input();
            nn_inference(in, nn_out);
            cls = nn_argmax(nn_out);
            nn_decode_output(nn_out, probs);

            if (cls == last_cls)
                stable++;
            else {
                stable = 1;
                last_cls = cls;
            }

            if (g_recording && stable >= 2
                && cls >= 0 && cls < NN_NUM_CLASSES) {
                g_record_stats.count[cls]++;
                g_record_stats.total_inferences++;
                g_record_stats.last_action = cls;
                g_record_stats.last_stable_action = cls;
            }

            rt_kprintf("[AI] %s %s (%d,%d,%d)\n",
                       g_recording ? "REC" : "---",
                       nn_class_names[cls],
                       (int)(probs[0] * 100),
                       (int)(probs[1] * 100),
                       (int)(probs[2] * 100));
        }
        rt_thread_mdelay(20);
    }
}

static int ai_task_init(void)
{
    rt_thread_t tid;
    tid = rt_thread_create("AI_NN", ai_thread_entry, RT_NULL, 8192, 16, 10);
    if (tid != RT_NULL) {
        rt_thread_startup(tid);
        rt_kprintf("[AI] Thread created\n");
    }
    return 0;
}
INIT_APP_EXPORT(ai_task_init);
