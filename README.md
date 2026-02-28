# Edgi-Talk IMU AI

**基于 RT-Thread 的嵌入式端到端 AI 动作识别系统**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![RT-Thread](https://img.shields.io/badge/RTOS-RT--Thread%205.x-brightgreen)](https://www.rt-thread.org/)
[![TensorFlow](https://img.shields.io/badge/Training-TensorFlow%202.x-orange)](https://www.tensorflow.org/)
[![LVGL](https://img.shields.io/badge/GUI-LVGL%209.x-red)](https://lvgl.io/)
[![Platform](https://img.shields.io/badge/Platform-PSoC%20Edge%20M55-purple)](https://www.infineon.com/)

中文 | [English](#introduction)

---

## 简介

本项目是一个完整的端到端嵌入式 AI 动作识别系统，基于 Edgi-Talk 开发板（Infineon PSoC Edge M55 应用核）和 RT-Thread 实时操作系统，使用板载 IMU（LSM6DS3 六轴传感器）实现三种动作的实时识别：

- **Stationary** — 静止（平放 / 手持不动）
- **Shaking** — 摇晃
- **Circle** — 画圈

---

## 系统架构

```text
 ① 数据采集               ② 模型训练               ③ 部署推理
+------------------+     +------------------+     +------------------+
| firmware/        |     | model-training/  |     | firmware/        |
| data-collector   | --> |                  | --> | ai-inference     |
|                  | CSV |                  | .c  |                  |
+------------------+     +------------------+     +------------------+
  Edgi-Talk 开发板          PC (Python)             Edgi-Talk 开发板
  USB 串口采集              TensorFlow 训练          实时 AI 推理
  LVGL 波形显示             INT8 量化 + C 代码       LVGL 结果展示
```

---

## 项目特色

- **纯 C 推理引擎** — 不依赖 TFLite Micro 或 C++，纯 C 实现 INT8 量化 CNN 推理
- **LVGL 图形界面** — 数据采集时展示实时波形，推理时展示动作识别统计
- **端到端流水线** — 数据采集、训练、量化、C 代码生成、嵌入式部署全流程覆盖
- **50Hz 实时推理** — 滑动窗口 100 采样点（2 秒），20ms 推理周期
- **超轻量模型** — 3 层 Conv2D + Dense，INT8 模型约 50KB，准确率 96.6%
- **自动化部署** — 训练完成后一键生成 C 代码并部署到固件工程

---

## 模型性能

```text
模型架构 ........... 3x Conv2D + BatchNorm + MaxPool + Dense
Float32 大小 ....... 182.7 KB
INT8 大小 .......... 50.6 KB
Float32 准确率 ..... 96.64%
INT8 准确率 ........ 96.64%
推理窗口 ........... 2 秒 (100 采样点 @ 50Hz)
窗口步进 ........... 1 秒 (50 采样点, 50% 重叠)
训练数据 ........... 2973 个窗口 (2 人采集)
```

模型架构如下：

```text
Input [100, 6, 1]
    |
    v
Conv2D(8, 5x1) + BatchNorm + ReLU + MaxPool(2x1)
    |
    v
Conv2D(16, 5x1) + BatchNorm + ReLU + MaxPool(2x1)
    |
    v
Conv2D(24, 3x3) + BatchNorm + ReLU + MaxPool(2x1)
    |
    v
Flatten -> Dense(24) + ReLU
    |
    v
Dense(3) + Softmax -> [stationary, shaking, circle]
```

---

## 项目结构

```text
Edgi-Talk-IMU-AI/
├── README.md
├── LICENSE
├── firmware/
│   ├── data-collector/                 # ① 数据采集固件
│   │   └── applications/
│   │       ├── main.c                  #    主程序 + LVGL UI (实时波形)
│   │       ├── imu_task.c              #    IMU 采集 + USB 串口发送
│   │       └── cdc_acm.c              #    USB CDC 虚拟串口驱动
│   └── ai-inference/                   # ③ AI 推理固件
│       └── applications/
│           ├── main.c                  #    主程序 + LVGL UI (识别结果)
│           ├── imu_task.c              #    IMU 六轴数据采集
│           ├── ai_task.c              #    AI 推理线程
│           ├── nn_model.c/h           #    神经网络 (自动生成)
│           └── preprocessing.c/h      #    预处理 (自动生成)
├── model-training/                     # ② 模型训练 (Python)
│   ├── run_training.py                 #    一键训练脚本
│   ├── train_model.ipynb              #    Jupyter Notebook 交互训练
│   ├── generate_c_inference.py        #    生成纯 C 推理引擎
│   ├── extract_model_weights.py       #    提取模型权重
│   ├── verify_c_inference.py          #    验证 C 推理正确性
│   ├── requirements.txt               #    Python 依赖
│   ├── Data/                           #    训练数据集 (2 人采集)
│   ├── output/                         #    训练输出 (模型 + C 代码)
│   └── Models/                         #    预训练模型
├── tools/
│   ├── record.py                       #    PC 端串口数据录入
│   └── deploy_model.py                #    一键部署脚本
└── docs/
    ├── hardware.md                     #    硬件接口说明
    └── figures/                        #    硬件接口图
```

固件工程中的 `rt-thread/`、`libraries/`、`packages/` 等第三方依赖未包含在仓库中，请参照下方环境搭建章节获取。

---

## 快速开始

### 环境要求

- **开发板** — Edgi-Talk (Infineon PSoC Edge, M55 应用核)
- **嵌入式 IDE** — RT-Thread Studio 2.x+
- **交叉编译器** — arm-none-eabi-gcc
- **Python** — 3.8+ (用于模型训练)
- **Python 依赖** — TensorFlow 2.12~2.16, NumPy, Pandas, scikit-learn

### 环境搭建

**固件开发环境：**

1. 安装 [RT-Thread Studio](https://www.rt-thread.org/studio.html)
2. 安装 arm-none-eabi-gcc 工具链（RT-Thread Studio 内置或手动安装）
3. 通过 RT-Thread Studio 包管理器获取 RT-Thread 内核 (v5.x)、LVGL (v9.x)、CMSIS 5
4. 获取 Infineon PSoC Edge SDK 库文件，放置到 `firmware/*/libraries/` 目录

**Python 训练环境：**

```bash
cd model-training/
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

### 第一步：数据采集

1. 用 RT-Thread Studio 打开 `firmware/data-collector/` 工程
2. 编译并烧录至 Edgi-Talk 开发板
3. USB 连接开发板至 PC
4. 运行数据录入脚本：

```bash
cd tools/
python record.py
```

5. 在开发板 LCD 上点击 Start Recording 开始录制
6. 执行目标动作（静止 / 摇晃 / 画圈），按 Ctrl+C 结束
7. 数据保存为 CSV 文件

### 第二步：模型训练

```bash
cd model-training/
python run_training.py
```

也可以使用 Jupyter Notebook 交互式训练：

```bash
jupyter notebook train_model.ipynb
```

训练完成后 `output/` 目录下生成：

- `model_int8.tflite` — INT8 量化模型
- `nn_model.c` / `nn_model.h` — 纯 C 推理引擎代码
- `preprocessing.c` / `preprocessing.h` — 预处理代码
- `training_report.json` — 训练报告

### 第三步：部署推理

自动部署（推荐）：

```bash
cd tools/
python deploy_model.py
```

手动部署：将以下文件从 `model-training/output/` 复制到 `firmware/ai-inference/applications/`：

- `nn_model.c`
- `nn_model.h`
- `preprocessing.c`
- `preprocessing.h`

然后用 RT-Thread Studio 打开 `firmware/ai-inference/` 工程，编译烧录。上电后 LCD 显示动作识别界面，点击 REC 按钮开始实时识别。

---

## 硬件平台

- **MCU** — PSoC Edge M55 (Cortex-M55 应用核)
- **IMU** — ST LSM6DS3 (6 轴加速度计 + 陀螺仪)
- **IMU 接口** — 软件 I2C (SCL: P8.0, SDA: P8.1)，地址 0x6A
- **显示屏** — MIPI DSI LCD，512 x 800 分辨率
- **USB** — USB 2.0 HS，CDC ACM 虚拟串口 (CherryUSB)
- **调试器** — 板载 DAP

详细硬件接口请参考 [docs/hardware.md](docs/hardware.md)。

---

## 技术栈

- **RTOS** — RT-Thread 5.x
- **GUI** — LVGL 9.x
- **AI 训练** — TensorFlow / Keras
- **AI 推理** — 纯 C INT8 量化推理引擎（无外部依赖）
- **USB** — CherryUSB (CDC ACM)
- **IMU 驱动** — 软件 I2C + LSM6DS3
- **构建系统** — SCons
- **交叉编译** — arm-none-eabi-gcc

---

## Introduction

This project is a complete end-to-end embedded AI motion recognition system built on the Edgi-Talk development board (Infineon PSoC Edge M55 core) with RT-Thread RTOS. It uses the onboard IMU (LSM6DS3, 6-axis accelerometer + gyroscope) to recognize three motion types in real time:

- **Stationary** — Still (on table / held steady)
- **Shaking** — Shaking motion
- **Circle** — Circular motion

### Key Features

- **Pure C Inference Engine** — No TFLite Micro or C++ dependency, standalone INT8 quantized CNN in pure C
- **LVGL GUI** — Real-time waveform during data collection, recognition stats during inference
- **End-to-End Pipeline** — Data collection, training, INT8 quantization, C code generation, embedded deployment
- **50Hz Real-time Inference** — Sliding window (100 samples / 2s window) with 20ms inference cycle
- **Ultra-lightweight** — 3x Conv2D + Dense, INT8 model ~50KB, 96.6% test accuracy

### Quick Start

1. **Collect Data** — Flash `firmware/data-collector/`, record IMU data via USB CDC
2. **Train Model** — Run `python run_training.py` in model-training directory
3. **Deploy** — Run `python deploy_model.py` in tools directory
4. **Run** — Flash `firmware/ai-inference/`, real-time motion recognition on LCD

See the Chinese sections above for detailed step-by-step instructions.

---

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [RT-Thread](https://www.rt-thread.org/) — Real-time operating system
- [LVGL](https://lvgl.io/) — Embedded graphics library
- [TensorFlow](https://www.tensorflow.org/) — Machine learning framework
- [CherryUSB](https://github.com/cherry-embedded/CherryUSB) — USB stack
- [Infineon DEEPCRAFT Studio](https://softwaretools.infineon.com/) — Original dataset source

## Contributing

Contributions are welcome! Please feel free to submit Issues and Pull Requests.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request
