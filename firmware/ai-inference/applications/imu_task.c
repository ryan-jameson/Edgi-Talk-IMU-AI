/* imu_task.c - IMU data acquisition (LSM6DS3, SW I2C, 50Hz) */
#include <rtthread.h>
#include <rtdevice.h>
#include <board.h>

#define SCL_PIN      GET_PIN(8, 0)
#define SDA_PIN      GET_PIN(8, 1)
#define LSM6DS3_ADDR 0x6A

volatile int16_t g_acc_x  = 0;
volatile int16_t g_acc_y  = 0;
volatile int16_t g_acc_z  = 0;
volatile int16_t g_gyro_x = 0;
volatile int16_t g_gyro_y = 0;
volatile int16_t g_gyro_z = 0;

static void i2c_delay(void)
{
    volatile rt_uint32_t i = 800;
    while (i--);
}

static void set_sda(int state)
{
    if (state)
        rt_pin_mode(SDA_PIN, PIN_MODE_INPUT);
    else {
        rt_pin_mode(SDA_PIN, PIN_MODE_OUTPUT);
        rt_pin_write(SDA_PIN, PIN_LOW);
    }
}

static void set_scl(int state)
{
    if (state)
        rt_pin_mode(SCL_PIN, PIN_MODE_INPUT);
    else {
        rt_pin_mode(SCL_PIN, PIN_MODE_OUTPUT);
        rt_pin_write(SCL_PIN, PIN_LOW);
    }
}

static int get_sda(void)
{
    rt_pin_mode(SDA_PIN, PIN_MODE_INPUT);
    return rt_pin_read(SDA_PIN);
}

static void i2c_start(void)
{
    set_sda(1); set_scl(1); i2c_delay();
    set_sda(0); i2c_delay();
    set_scl(0); i2c_delay();
}

static void i2c_stop(void)
{
    set_sda(0); i2c_delay();
    set_scl(1); i2c_delay();
    set_sda(1); i2c_delay();
}

static int i2c_write_byte(rt_uint8_t data)
{
    int i, ack;
    for (i = 0; i < 8; i++) {
        set_sda((data & 0x80) ? 1 : 0);
        i2c_delay();
        set_scl(1); i2c_delay();
        set_scl(0); i2c_delay();
        data <<= 1;
    }
    set_sda(1); i2c_delay();
    set_scl(1); i2c_delay();
    ack = get_sda();
    set_scl(0); i2c_delay();
    return (ack == 0);
}

static rt_uint8_t i2c_read_byte(int send_ack)
{
    rt_uint8_t data = 0;
    int i;
    set_sda(1); i2c_delay();
    for (i = 0; i < 8; i++) {
        data <<= 1;
        set_scl(1); i2c_delay();
        if (get_sda()) data |= 1;
        set_scl(0); i2c_delay();
    }
    set_sda(send_ack ? 0 : 1);
    i2c_delay();
    set_scl(1); i2c_delay();
    set_scl(0); i2c_delay();
    set_sda(1); i2c_delay();
    return data;
}

static int imu_write_reg(rt_uint8_t reg, rt_uint8_t val)
{
    i2c_start();
    if (!i2c_write_byte((LSM6DS3_ADDR << 1) | 0)) {
        i2c_stop();
        return -1;
    }
    i2c_write_byte(reg);
    i2c_write_byte(val);
    i2c_stop();
    return 0;
}

static int imu_read_regs(rt_uint8_t reg, rt_uint8_t *buf, int len)
{
    int i;
    i2c_start();
    if (!i2c_write_byte((LSM6DS3_ADDR << 1) | 0)) {
        i2c_stop();
        return -1;
    }
    i2c_write_byte(reg);
    i2c_start();
    if (!i2c_write_byte((LSM6DS3_ADDR << 1) | 1)) {
        i2c_stop();
        return -1;
    }
    for (i = 0; i < len; i++)
        buf[i] = i2c_read_byte(i < len - 1);
    i2c_stop();
    return 0;
}

static void imu_thread_entry(void *parameter)
{
    rt_uint8_t buf[12];
    (void)parameter;
    imu_write_reg(0x10, 0x40);
    imu_write_reg(0x11, 0x44);

    while (1) {
        if (imu_read_regs(0x22, buf, 12) == 0) {
            g_gyro_x = (int16_t)((buf[1] << 8) | buf[0]);
            g_gyro_y = (int16_t)((buf[3] << 8) | buf[2]);
            g_gyro_z = (int16_t)((buf[5] << 8) | buf[4]);
            g_acc_x  = (int16_t)((buf[7] << 8) | buf[6]);
            g_acc_y  = (int16_t)((buf[9] << 8) | buf[8]);
            g_acc_z  = (int16_t)((buf[11] << 8) | buf[10]);
        }
        rt_thread_mdelay(20);
    }
}

static int imu_task_init(void)
{
    rt_thread_t tid;
    tid = rt_thread_create("imu", imu_thread_entry, RT_NULL, 2048, 20, 10);
    if (tid != RT_NULL)
        rt_thread_startup(tid);
    return 0;
}
INIT_APP_EXPORT(imu_task_init);
