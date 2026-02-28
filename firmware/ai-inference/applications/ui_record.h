#ifndef UI_RECORD_H
#define UI_RECORD_H

#include <rtthread.h>
#include "nn_model.h"

extern volatile int g_recording;

typedef struct {
    volatile rt_uint32_t count[NN_NUM_CLASSES];
    volatile rt_uint32_t total_inferences;
    volatile int         last_action;
    volatile int         last_stable_action;
} record_stats_t;

extern record_stats_t g_record_stats;

static inline void record_stats_reset(void)
{
    int i;
    for (i = 0; i < NN_NUM_CLASSES; i++)
        g_record_stats.count[i] = 0;
    g_record_stats.total_inferences = 0;
    g_record_stats.last_action = -1;
    g_record_stats.last_stable_action = -1;
}

#endif
