#include <rtthread.h>
#include <rtdevice.h>
#include <board.h>
#include "lv_port_disp.h"
#include "lvgl.h"

#define LED_PIN_G GET_PIN(16, 6)

/* 引用外部变量 */
extern volatile int g_is_recording;
extern volatile int16_t g_acc_x;
extern volatile int16_t g_acc_y;
extern volatile int16_t g_acc_z;

lv_obj_t * ui_label_status;
lv_obj_t * ui_btn;
lv_obj_t * ui_btn_label;
lv_obj_t * ui_chart;
lv_chart_series_t * ser_x;
lv_chart_series_t * ser_y;
lv_chart_series_t * ser_z;

/* 定时刷新波形图 */
static void chart_update_timer_cb(lv_timer_t * timer) {
    if (g_is_recording) {
        lv_chart_set_next_value(ui_chart, ser_x, g_acc_x);
        lv_chart_set_next_value(ui_chart, ser_y, g_acc_y);
        lv_chart_set_next_value(ui_chart, ser_z, g_acc_z);
    }
}

/* 按钮点击 Toggle 逻辑 */
static void btn_event_cb(lv_event_t * e) {
    lv_event_code_t code = lv_event_get_code(e);
    if(code == LV_EVENT_CLICKED) {
        if (g_is_recording == 0) {
            g_is_recording = 1;
            lv_label_set_text(ui_label_status, "Recording & Plotting...");
            lv_label_set_text(ui_btn_label, "Stop Recording");
            lv_obj_set_style_bg_color(ui_btn, lv_color_hex(0xE74C3C), 0);
        } else {
            g_is_recording = 0;
            lv_label_set_text(ui_label_status, "Waiting for action...");
            lv_label_set_text(ui_btn_label, "Start Recording");
            lv_obj_set_style_bg_color(ui_btn, lv_color_hex(0x3498DB), 0);
        }
    }
}

/* UI 搭建 */
void my_gui_setup(void) {
    lv_obj_t * scr = lv_screen_active();
    lv_obj_set_style_bg_color(scr, lv_color_hex(0x222222), 0);
    lv_obj_set_style_text_color(scr, lv_color_hex(0xFFFFFF), 0);

    ui_label_status = lv_label_create(scr);
    lv_label_set_text(ui_label_status, "Waiting for action...");
    lv_obj_align(ui_label_status, LV_ALIGN_TOP_MID, 0, 40);

    ui_btn = lv_button_create(scr);
    lv_obj_set_size(ui_btn, 200, 80);
    lv_obj_align(ui_btn, LV_ALIGN_TOP_MID, 0, 100);
    lv_obj_set_style_bg_color(ui_btn, lv_color_hex(0x3498DB), 0);
    lv_obj_add_event_cb(ui_btn, btn_event_cb, LV_EVENT_ALL, NULL);

    ui_btn_label = lv_label_create(ui_btn);
    lv_label_set_text(ui_btn_label, "Start Recording");
    lv_obj_center(ui_btn_label);

    ui_chart = lv_chart_create(scr);
    lv_obj_set_size(ui_chart, 400, 300);
    lv_obj_align(ui_chart, LV_ALIGN_BOTTOM_MID, 0, -80);
    lv_chart_set_type(ui_chart, LV_CHART_TYPE_LINE);
    lv_chart_set_point_count(ui_chart, 100);
    lv_chart_set_update_mode(ui_chart, LV_CHART_UPDATE_MODE_SHIFT);
    lv_chart_set_range(ui_chart, LV_CHART_AXIS_PRIMARY_Y, -32000, 32000);

    ser_x = lv_chart_add_series(ui_chart, lv_palette_main(LV_PALETTE_RED), LV_CHART_AXIS_PRIMARY_Y);
    ser_y = lv_chart_add_series(ui_chart, lv_palette_main(LV_PALETTE_GREEN), LV_CHART_AXIS_PRIMARY_Y);
    ser_z = lv_chart_add_series(ui_chart, lv_palette_main(LV_PALETTE_BLUE), LV_CHART_AXIS_PRIMARY_Y);

    lv_timer_create(chart_update_timer_cb, 20, NULL);
}

void lv_user_gui_init(void) {
    my_gui_setup();
}

int main(void) {
    rt_kprintf("M55 LVGL + IMU System Init\n");
    rt_pin_mode(LED_PIN_G, PIN_MODE_OUTPUT);

    /* 启动 LVGL (USB和IMU线程已通过宏自动启动) */
    lvgl_thread_init();

    while (1) {
        rt_pin_write(LED_PIN_G, PIN_LOW);
        rt_thread_mdelay(500);
        rt_pin_write(LED_PIN_G, PIN_HIGH);
        rt_thread_mdelay(500);
    }
    return 0;
}
