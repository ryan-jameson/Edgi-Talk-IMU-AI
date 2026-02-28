# AI 推理固件 — Edgi-Talk IMU 动作识别

[**中文**](#简介) | [**English**](#introduction)

## 简介

本工程是 Edgi-Talk IMU AI 项目的 **推理部署固件**，运行在 RT-Thread 实时操作系统上。  
它使用板载 IMU（LSM6DS3）采集六轴数据，通过 **纯 C INT8 量化 CNN** 实现三种动作的实时识别，并在 LCD 上通过 LVGL 展示识别结果。

### 功能模块

| 文件 | 功能 |
|------|------|
| `applications/main.c` | 系统入口 + LVGL UI（动作识别界面、统计柱状图、实时趋势图） |
| `applications/imu_task.c` | IMU 数据采集线程（LSM6DS3 六轴，50Hz） |
| `applications/ai_task.c` | AI 推理线程（纯 C INT8 CNN，滑动窗口 2s） |
| `applications/nn_model.c/h` | 神经网络模型权重与推理函数（由训练脚本自动生成） |
| `applications/preprocessing.c/h` | 输入预处理（归一化 + INT8 量化，由训练脚本自动生成） |
| `applications/ui_record.h` | UI 与推理线程共享的统计数据结构 |

### 编译与烧录

1. 获取 RT-Thread 内核与 SDK 库（参见根目录 README 中的环境搭建部分）
2. 使用 RT-Thread Studio 或 SCons 编译本工程
3. 通过板载 DAP 下载器烧录至 Edgi-Talk 开发板
4. 上电后 LCD 显示动作识别界面，点击 **REC** 按钮开始实时推理

---

## Introduction

This is the **AI inference firmware** of the Edgi-Talk IMU AI project, running on RT-Thread RTOS.  
It reads 6-axis IMU data (LSM6DS3), runs a **pure C INT8 quantized CNN** for real-time motion recognition, and displays results on LCD via LVGL.

### Build & Flash

1. Obtain RT-Thread kernel and SDK libraries (see root README for setup)
2. Build with RT-Thread Studio or SCons
3. Flash via onboard DAP debugger
4. After power-on, press **REC** on the LCD to start real-time inference
