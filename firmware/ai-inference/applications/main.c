/* main.c - System entry + LVGL UI (512x800) */
#include <rtthread.h>
#include <rtdevice.h>
#include <board.h>
#include "lvgl.h"
#include "nn_model.h"
#include "ui_record.h"

#define LED_PIN_G GET_PIN(16, 6)

volatile int g_recording = 0;
record_stats_t g_record_stats = {{0}, 0, -1, -1};

static lv_obj_t *ui_label_title;
static lv_obj_t *ui_label_action;
static lv_obj_t *ui_label_status;
static lv_obj_t *ui_btn_record;
static lv_obj_t *ui_btn_label;
static lv_obj_t *ui_bar_stat[3];
static lv_obj_t *ui_label_stat[3];
static lv_obj_t *ui_label_pct[3];
static lv_obj_t *ui_label_total;
static lv_obj_t *ui_chart;
static lv_chart_series_t *ui_chart_ser[3];

static const char *ACT_NAME[3] = {"Stationary", "Shaking", "Circle"};
static const lv_palette_t ACT_PAL[3] = {
    LV_PALETTE_GREY, LV_PALETTE_RED, LV_PALETTE_GREEN
};

static int c_action = -1;
static rt_uint32_t c_total = 0;
static int c_rec = -1;

void update_ai_label(const char *text, int user_id)
{
    (void)text;
    (void)user_id;
}

static void btn_record_cb(lv_event_t *e)
{
    int i;
    if (lv_event_get_code(e) != LV_EVENT_CLICKED)
        return;
    if (g_recording) {
        g_recording = 0;
        rt_kprintf("[UI] Stop total=%d\n",
                   (int)g_record_stats.total_inferences);
    } else {
        record_stats_reset();
        c_action = -1;
        c_total = 0;
        lv_label_set_text(ui_label_action, "Detecting...");
        lv_obj_set_style_text_color(ui_label_action,
                                    lv_color_hex(0xFFFFFF), 0);
        for (i = 0; i < 3; i++) {
            lv_bar_set_value(ui_bar_stat[i], 0, LV_ANIM_OFF);
            lv_label_set_text(ui_label_pct[i], "0%  (0)");
            lv_chart_set_all_value(ui_chart, ui_chart_ser[i], 0);
        }
        lv_label_set_text(ui_label_total, "Total: 0");
        lv_chart_refresh(ui_chart);
        g_recording = 1;
        rt_kprintf("[UI] Recording\n");
    }
}

static void ui_refresh_cb(lv_timer_t *timer)
{
    int rec, aid, i, pct;
    rt_uint32_t total, cnt;
    char buf[48];

    (void)timer;
    rec = g_recording;
    aid = g_record_stats.last_stable_action;
    total = g_record_stats.total_inferences;

    if (rec != c_rec) {
        c_rec = rec;
        if (rec) {
            lv_label_set_text(ui_btn_label, LV_SYMBOL_STOP " STOP");
            lv_obj_set_style_bg_color(ui_btn_record,
                lv_palette_main(LV_PALETTE_RED), 0);
            lv_label_set_text(ui_label_status, "Recording...");
            lv_obj_set_style_text_color(ui_label_status,
                lv_palette_main(LV_PALETTE_RED), 0);
        } else {
            lv_label_set_text(ui_btn_label, LV_SYMBOL_PLAY " REC");
            lv_obj_set_style_bg_color(ui_btn_record,
                lv_palette_main(LV_PALETTE_GREEN), 0);
            lv_label_set_text(ui_label_status,
                (c_action >= 0) ? "Stopped" : "Ready");
            lv_obj_set_style_text_color(ui_label_status,
                lv_palette_main(LV_PALETTE_GREY), 0);
        }
    }

    if (rec && aid != c_action && aid >= 0 && aid < 3) {
        c_action = aid;
        lv_label_set_text(ui_label_action, ACT_NAME[aid]);
        lv_obj_set_style_text_color(ui_label_action,
            lv_palette_main(ACT_PAL[aid]), 0);
    }

    if (rec && total != c_total) {
        c_total = total;
        for (i = 0; i < 3; i++) {
            cnt = g_record_stats.count[i];
            pct = (total > 0) ? (int)((cnt * 100) / total) : 0;
            lv_bar_set_value(ui_bar_stat[i], pct, LV_ANIM_ON);
            rt_snprintf(buf, sizeof(buf), "%d%%  (%d)", pct, (int)cnt);
            lv_label_set_text(ui_label_pct[i], buf);
            lv_chart_set_next_value(ui_chart, ui_chart_ser[i],
                                    (int32_t)cnt);
        }
        rt_snprintf(buf, sizeof(buf), "Total: %d", (int)total);
        lv_label_set_text(ui_label_total, buf);
        lv_chart_refresh(ui_chart);
    }
}

static void my_gui_setup(void)
{
    int i, y;
    lv_obj_t *scr = lv_screen_active();
    lv_obj_set_style_bg_color(scr, lv_color_hex(0x1A1A2E), 0);

    ui_label_title = lv_label_create(scr);
    lv_label_set_text(ui_label_title, "IMU Movement Detection");
    lv_obj_set_style_text_color(ui_label_title, lv_color_hex(0x8888AA), 0);
    lv_obj_set_style_text_font(ui_label_title, &lv_font_montserrat_16, 0);
    lv_obj_align(ui_label_title, LV_ALIGN_TOP_MID, 0, 15);

    ui_label_action = lv_label_create(scr);
    lv_label_set_text(ui_label_action, "Ready");
    lv_obj_set_style_text_font(ui_label_action, &lv_font_montserrat_24, 0);
    lv_obj_set_style_text_color(ui_label_action, lv_color_hex(0xFFFFFF), 0);
    lv_obj_align(ui_label_action, LV_ALIGN_TOP_MID, 0, 60);

    ui_label_status = lv_label_create(scr);
    lv_label_set_text(ui_label_status, "Ready");
    lv_obj_set_style_text_color(ui_label_status,
                                lv_palette_main(LV_PALETTE_GREY), 0);
    lv_obj_set_style_text_font(ui_label_status, &lv_font_montserrat_14, 0);
    lv_obj_align(ui_label_status, LV_ALIGN_TOP_MID, 0, 95);

    ui_btn_record = lv_button_create(scr);
    lv_obj_set_size(ui_btn_record, 200, 50);
    lv_obj_align(ui_btn_record, LV_ALIGN_TOP_MID, 0, 130);
    lv_obj_set_style_bg_color(ui_btn_record,
                              lv_palette_main(LV_PALETTE_GREEN), 0);
    lv_obj_set_style_radius(ui_btn_record, 25, 0);
    lv_obj_set_style_shadow_width(ui_btn_record, 8, 0);
    lv_obj_set_style_shadow_color(ui_btn_record,
                                  lv_color_hex(0x000000), 0);
    lv_obj_set_style_shadow_opa(ui_btn_record, LV_OPA_50, 0);
    lv_obj_add_event_cb(ui_btn_record, btn_record_cb,
                        LV_EVENT_CLICKED, NULL);

    ui_btn_label = lv_label_create(ui_btn_record);
    lv_label_set_text(ui_btn_label, LV_SYMBOL_PLAY " REC");
    lv_obj_set_style_text_font(ui_btn_label, &lv_font_montserrat_20, 0);
    lv_obj_set_style_text_color(ui_btn_label, lv_color_hex(0xFFFFFF), 0);
    lv_obj_center(ui_btn_label);

    {
        lv_obj_t *sep = lv_obj_create(scr);
        lv_obj_set_size(sep, 460, 2);
        lv_obj_set_style_bg_color(sep, lv_color_hex(0x333355), 0);
        lv_obj_set_style_border_width(sep, 0, 0);
        lv_obj_set_style_radius(sep, 0, 0);
        lv_obj_align(sep, LV_ALIGN_TOP_MID, 0, 200);
    }

    {
        lv_obj_t *st = lv_label_create(scr);
        lv_label_set_text(st, "Action Statistics");
        lv_obj_set_style_text_color(st, lv_color_hex(0x8888AA), 0);
        lv_obj_set_style_text_font(st, &lv_font_montserrat_14, 0);
        lv_obj_align(st, LV_ALIGN_TOP_MID, 0, 215);
    }

    for (i = 0; i < 3; i++) {
        y = 245 + i * 55;
        ui_label_stat[i] = lv_label_create(scr);
        lv_label_set_text(ui_label_stat[i], ACT_NAME[i]);
        lv_obj_set_style_text_color(ui_label_stat[i],
            lv_palette_main(ACT_PAL[i]), 0);
        lv_obj_set_style_text_font(ui_label_stat[i],
            &lv_font_montserrat_14, 0);
        lv_obj_align(ui_label_stat[i], LV_ALIGN_TOP_LEFT, 20, y);

        ui_bar_stat[i] = lv_bar_create(scr);
        lv_obj_set_size(ui_bar_stat[i], 260, 18);
        lv_obj_align(ui_bar_stat[i], LV_ALIGN_TOP_LEFT, 130, y + 2);
        lv_bar_set_range(ui_bar_stat[i], 0, 100);
        lv_bar_set_value(ui_bar_stat[i], 0, LV_ANIM_OFF);
        lv_obj_set_style_bg_color(ui_bar_stat[i],
            lv_color_hex(0x333355), 0);
        lv_obj_set_style_bg_color(ui_bar_stat[i],
            lv_palette_main(ACT_PAL[i]), LV_PART_INDICATOR);
        lv_obj_set_style_radius(ui_bar_stat[i], 4, 0);
        lv_obj_set_style_radius(ui_bar_stat[i], 4, LV_PART_INDICATOR);

        ui_label_pct[i] = lv_label_create(scr);
        lv_label_set_text(ui_label_pct[i], "0%  (0)");
        lv_obj_set_style_text_color(ui_label_pct[i],
            lv_color_hex(0xBBBBBB), 0);
        lv_obj_set_style_text_font(ui_label_pct[i],
            &lv_font_montserrat_12, 0);
        lv_obj_align(ui_label_pct[i], LV_ALIGN_TOP_LEFT, 400, y + 3);
    }

    ui_label_total = lv_label_create(scr);
    lv_label_set_text(ui_label_total, "Total: 0");
    lv_obj_set_style_text_color(ui_label_total,
                                lv_color_hex(0x999999), 0);
    lv_obj_set_style_text_font(ui_label_total,
                               &lv_font_montserrat_14, 0);
    lv_obj_align(ui_label_total, LV_ALIGN_TOP_MID, 0, 415);

    {
        lv_obj_t *sep2 = lv_obj_create(scr);
        lv_obj_set_size(sep2, 460, 2);
        lv_obj_set_style_bg_color(sep2, lv_color_hex(0x333355), 0);
        lv_obj_set_style_border_width(sep2, 0, 0);
        lv_obj_set_style_radius(sep2, 0, 0);
        lv_obj_align(sep2, LV_ALIGN_TOP_MID, 0, 445);
    }

    {
        lv_obj_t *ct = lv_label_create(scr);
        lv_label_set_text(ct, "Detection Timeline");
        lv_obj_set_style_text_color(ct, lv_color_hex(0x8888AA), 0);
        lv_obj_set_style_text_font(ct, &lv_font_montserrat_14, 0);
        lv_obj_align(ct, LV_ALIGN_TOP_MID, 0, 458);
    }

    ui_chart = lv_chart_create(scr);
    lv_obj_set_size(ui_chart, 460, 280);
    lv_obj_align(ui_chart, LV_ALIGN_TOP_MID, 0, 485);
    lv_chart_set_type(ui_chart, LV_CHART_TYPE_LINE);
    lv_chart_set_point_count(ui_chart, 30);
    lv_chart_set_range(ui_chart, LV_CHART_AXIS_PRIMARY_Y, 0, 50);
    lv_chart_set_div_line_count(ui_chart, 4, 0);
    lv_chart_set_update_mode(ui_chart, LV_CHART_UPDATE_MODE_SHIFT);
    lv_obj_set_style_bg_color(ui_chart, lv_color_hex(0x0F0F23), 0);
    lv_obj_set_style_border_color(ui_chart,
                                  lv_color_hex(0x333355), 0);
    lv_obj_set_style_border_width(ui_chart, 1, 0);
    lv_obj_set_style_radius(ui_chart, 8, 0);
    lv_obj_set_style_line_color(ui_chart,
                                lv_color_hex(0x222244), LV_PART_MAIN);
    lv_obj_set_style_pad_all(ui_chart, 10, 0);

    for (i = 0; i < 3; i++) {
        ui_chart_ser[i] = lv_chart_add_series(ui_chart,
            lv_palette_main(ACT_PAL[i]), LV_CHART_AXIS_PRIMARY_Y);
        lv_chart_set_all_value(ui_chart, ui_chart_ser[i], 0);
    }

    lv_timer_create(ui_refresh_cb, 200, NULL);
}

void lv_user_gui_init(void)
{
    my_gui_setup();
}

int main(void)
{
    rt_kprintf("System Start: PSoC Edge AI\n");
    rt_pin_mode(LED_PIN_G, PIN_MODE_OUTPUT);
    lvgl_thread_init();

    while (1) {
        rt_pin_write(LED_PIN_G, PIN_LOW);
        rt_thread_mdelay(100);
        rt_pin_write(LED_PIN_G, PIN_HIGH);
        rt_thread_mdelay(900);
    }
    return 0;
}
