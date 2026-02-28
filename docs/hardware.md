# Edgi-Talk IMU AI — 硬件接口说明

## 开发板概述

本项目使用 **Edgi-Talk 开发板**，基于 Infineon PSoC Edge 系列芯片，具备双核架构（M33 + M55）。本项目运行在 **M55 应用核**（Cortex-M55 / Cortex-M7 兼容模式）上。

## IMU 传感器

- **型号**：ST LSM6DS3
- **接口**：软件模拟 I2C
  - SCL: `P8.0` (GPIO Pin 8,0)
  - SDA: `P8.1` (GPIO Pin 8,1)
- **I2C 地址**：`0x6A`
- **采样率**：104Hz（固件内 50Hz 读取频率）
- **量程配置**：
  - 加速度计: `CTRL1_XL (0x10) = 0x40` → 104Hz, ±2g
  - 陀螺仪: `CTRL2_G (0x11) = 0x44` → 104Hz, ±500dps
- **灵敏度**：
  - 加速度计: 0.000598 g/LSB
  - 陀螺仪: 0.0175 dps/LSB

## 显示屏

- **接口**：MIPI DSI
- **分辨率**：512 × 800
- **GUI 框架**：LVGL 9.x

## USB 接口

- **协议**：USB 2.0 High Speed
- **功能**：CDC ACM 虚拟串口（用于数据采集阶段）
- **驱动**：CherryUSB
- **端点配置**：
  - IN Endpoint: `0x81`
  - OUT Endpoint: `0x02`
  - INT Endpoint: `0x83`

## 硬件接口图

### 背光接口
![背光接口](figures/1.png)

### MIPI 接口
![MIPI接口](figures/2.png)

### 电源接口
![电源接口](figures/3.png)

### BTB 座子
![BTB座子](figures/4.png)
![BTB座子](figures/5.png)

### MCU 接口
![MCU接口](figures/6.png)
![MCU接口](figures/7.png)
