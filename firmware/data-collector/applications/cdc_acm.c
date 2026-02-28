#include "usbd_core.h"
#include "usbd_cdc_acm.h"
#include <rtthread.h>

#define CDC_IN_EP  0x81
#define CDC_OUT_EP 0x02
#define CDC_INT_EP 0x83

static const uint8_t device_descriptor[] = {
    USB_DEVICE_DESCRIPTOR_INIT(USB_2_0, 0xEF, 0x02, 0x01, 0xFFFF, 0xFFFF, 0x0100, 0x01)
};
static const uint8_t config_descriptor[] = {
    USB_CONFIG_DESCRIPTOR_INIT((9 + CDC_ACM_DESCRIPTOR_LEN), 0x02, 0x01, USB_CONFIG_BUS_POWERED, 100),
    CDC_ACM_DESCRIPTOR_INIT(0x00, CDC_INT_EP, CDC_OUT_EP, CDC_IN_EP, 512, 0x02)
};
static const uint8_t device_quality_descriptor[] = {
    0x0a, USB_DESCRIPTOR_TYPE_DEVICE_QUALIFIER, 0x00, 0x02, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00,
};
static const char *string_descriptors[] = {
    (const char[]){ 0x09, 0x04 }, "CherryUSB", "Edgi-Talk IMU AI", "20260223",
};

static const uint8_t *device_descriptor_callback(uint8_t speed) { return device_descriptor; }
static const uint8_t *config_descriptor_callback(uint8_t speed) { return config_descriptor; }
static const uint8_t *device_quality_descriptor_callback(uint8_t speed) { return device_quality_descriptor; }
static const char *string_descriptor_callback(uint8_t speed, uint8_t index) {
    if (index > 3) return NULL; return string_descriptors[index];
}

const struct usb_descriptor cdc_descriptor = {
    .device_descriptor_callback = device_descriptor_callback,
    .config_descriptor_callback = config_descriptor_callback,
    .device_quality_descriptor_callback = device_quality_descriptor_callback,
    .string_descriptor_callback = string_descriptor_callback
};

volatile bool ep_tx_busy_flag = false;
volatile bool usb_is_configured = false;

static void usbd_event_handler(uint8_t busid, uint8_t event) {
    if (event == USBD_EVENT_CONFIGURED) { usb_is_configured = true; ep_tx_busy_flag = false; }
    else if (event == USBD_EVENT_DISCONNECTED) { usb_is_configured = false; }
}
void usbd_cdc_acm_bulk_out(uint8_t busid, uint8_t ep, uint32_t nbytes) { usbd_ep_start_read(busid, CDC_OUT_EP, NULL, 0); }
void usbd_cdc_acm_bulk_in(uint8_t busid, uint8_t ep, uint32_t nbytes) {
    if ((nbytes % usbd_get_ep_mps(busid, ep)) == 0 && nbytes) usbd_ep_start_write(busid, CDC_IN_EP, NULL, 0);
    else ep_tx_busy_flag = false;
}

struct usbd_endpoint cdc_out_ep = { .ep_addr = CDC_OUT_EP, .ep_cb = usbd_cdc_acm_bulk_out };
struct usbd_endpoint cdc_in_ep = { .ep_addr = CDC_IN_EP, .ep_cb = usbd_cdc_acm_bulk_in };
static struct usbd_interface intf0, intf1;

/* >>> 暴露给外部的发送函数 <<< */
int my_usb_send_data(uint8_t *data, uint32_t len) {
    if(!usb_is_configured) return -1;
    int timeout = 1000;
    while (ep_tx_busy_flag && timeout > 0) { rt_thread_mdelay(1); timeout--; }
    if (timeout == 0) return -2;
    ep_tx_busy_flag = true;
    return usbd_ep_start_write(0, CDC_IN_EP, data, len);
}

static int cherryusb_cdc_init(void) {
    usbd_desc_register(0, &cdc_descriptor);
    usbd_add_interface(0, usbd_cdc_acm_init_intf(0, &intf0));
    usbd_add_interface(0, usbd_cdc_acm_init_intf(0, &intf1));
    usbd_add_endpoint(0, &cdc_out_ep);
    usbd_add_endpoint(0, &cdc_in_ep);
    usbd_initialize(0, USBHS_BASE, usbd_event_handler);
    return 0;
}
INIT_APP_EXPORT(cherryusb_cdc_init);
