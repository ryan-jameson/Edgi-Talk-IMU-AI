# 数据采集固件 — Edgi-Talk IMU 数据录制

[**中文**](#简介) | [**English**](#introduction)

## 简介

本工程是 Edgi-Talk IMU AI 项目的 **数据采集固件**，运行在 RT-Thread 实时操作系统上。  
它使用板载 IMU（LSM6DS3）以 50Hz 采集加速度数据，通过 USB CDC 虚拟串口发送至 PC，同时在 LCD 上通过 LVGL 实时展示波形。

### 功能模块

| 文件 | 功能 |
|------|------|
| `applications/main.c` | 系统入口 + LVGL UI（实时波形显示、录制控制按钮） |
| `applications/imu_task.c` | IMU 数据采集 + USB 串口发送线程（50Hz） |
| `applications/cdc_acm.c` | USB CDC ACM 虚拟串口驱动（CherryUSB） |

### 使用方法

1. 编译并烧录本固件至 Edgi-Talk 开发板
2. 用 USB 数据线连接开发板至 PC
3. 在 PC 端运行 `tools/record.py` 监听串口数据
4. 在 LCD 屏幕上点击 **Start Recording** 按钮开始录制
5. 执行目标动作（静止 / 摇晃 / 画圈），完成后按 Ctrl+C 结束
6. 数据保存为 CSV 文件，可用于模型训练

### 数据格式

USB 串口输出 CSV 格式：

```
AccX,AccY,AccZ
-123,456,16384
...
```

每行三个加速度计原始值（int16），以逗号分隔，采样率 50Hz。

---

## Introduction

This is the **data collection firmware** of the Edgi-Talk IMU AI project, running on RT-Thread RTOS.  
It samples IMU data (LSM6DS3) at 50Hz, streams via USB CDC to PC, and displays real-time waveforms on LCD via LVGL.

### Usage

1. Build and flash this firmware to the Edgi-Talk board
2. Connect the board to PC via USB
3. Run `tools/record.py` on PC to record data
4. Tap **Start Recording** on the LCD to begin
5. Perform target motions (stationary / shaking / circle), press Ctrl+C to stop
6. Data is saved as CSV for model training
