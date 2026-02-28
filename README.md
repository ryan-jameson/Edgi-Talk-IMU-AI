<div align="center"># Edgi-Talk IMU AI — 基于 RT-Thread 的嵌入式动作识别系统



# 🧠 Edgi-Talk IMU AI[**中文**](#简介) | [**English**](#introduction)



### 基于 RT-Thread 的嵌入式端到端 AI 动作识别系统---



[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)## 简介

[![RT-Thread](https://img.shields.io/badge/RTOS-RT--Thread%205.x-brightgreen)](https://www.rt-thread.org/)

[![TensorFlow](https://img.shields.io/badge/Training-TensorFlow%202.x-orange)](https://www.tensorflow.org/)本项目是一个完整的 **端到端嵌入式 AI 动作识别系统**，基于 **Edgi-Talk 开发板**（PSoC Edge M55 应用核）和 **RT-Thread 实时操作系统**，使用板载 IMU（LSM6DS3 六轴加速度计 + 陀螺仪）实现三种动作的实时识别：

[![LVGL](https://img.shields.io/badge/GUI-LVGL%209.x-red)](https://lvgl.io/)

[![Platform](https://img.shields.io/badge/Platform-PSoC%20Edge%20M55-purple)](https://www.infineon.com/)| 动作类别 | 说明 |

|---------|------|

**纯 C 推理引擎 · INT8 量化 CNN · LVGL 图形界面 · 全流程覆盖**| **Stationary** | 静止（平放/手持不动） |

| **Shaking** | 摇晃 |

[中文](#-简介) · [English](#-introduction) · [快速开始](#-快速开始) · [项目结构](#-项目结构)| **Circle** | 画圈 |



</div>### 系统架构



---```

┌─────────────────────────────────────────────────────────┐

## ✨ 简介│                    完整工作流                              │

│                                                         │

本项目是一个完整的 **端到端嵌入式 AI 动作识别系统**，基于 **Edgi-Talk 开发板**（Infineon PSoC Edge M55 应用核）和 **RT-Thread 实时操作系统**，使用板载 IMU（LSM6DS3 六轴加速度计 + 陀螺仪）实现三种动作的实时识别：│  ① 数据采集        ② 模型训练         ③ 部署推理          │

│  ┌───────────┐    ┌──────────────┐   ┌──────────────┐   │

| 动作类别 | 说明 | 图标 |│  │ firmware/  │    │   model-     │   │  firmware/   │   │

|:--------:|------|:----:|│  │  data-     │───>│  training/   │──>│  ai-         │   │

| **Stationary** | 静止（平放 / 手持不动） | 🟡 |│  │ collector  │CSV │              │.c │ inference    │   │

| **Shaking** | 摇晃 | 🔴 |│  └───────────┘    └──────────────┘   └──────────────┘   │

| **Circle** | 画圈 | 🟢 |│   Edgi-Talk板       PC (Python)        Edgi-Talk板       │

│   USB串口采集        TensorFlow训练      实时AI推理        │

## 🏗️ 系统架构│   LVGL波形显示      INT8量化+C代码生成   LVGL结果展示       │

└─────────────────────────────────────────────────────────┘

``````

┌──────────────────────────────────────────────────────────────────┐

│                       完整工作流 Pipeline                         │### 项目特色

│                                                                  │

│   ① 数据采集             ② 模型训练              ③ 部署推理       │- 🎯 **纯 C 推理引擎**：不依赖 TFLite Micro / C++，纯 C 实现 INT8 量化 CNN 推理，适合资源受限 MCU

│   ┌─────────────┐       ┌──────────────┐       ┌──────────────┐ │- 📊 **LVGL 图形界面**：数据采集阶段展示实时波形，推理阶段展示动作识别统计

│   │  firmware/   │       │   model-     │       │  firmware/   │ │- 🔄 **端到端流水线**：从数据采集 → 模型训练 → 量化 → C代码生成 → 嵌入式部署，全流程覆盖

│   │  data-       │──CSV──│  training/   │──.c──▸│  ai-         │ │- ⚡ **50Hz 实时推理**：使用滑动窗口（100 采样点 / 2 秒窗口）实现实时动作分类

│   │  collector   │       │              │       │  inference   │ │- 🧠 **轻量级 CNN**：3 层 Conv2D + Dense，INT8 模型仅 ~50KB，测试准确率 96.6%

│   └─────────────┘       └──────────────┘       └──────────────┘ │

│    Edgi-Talk 板           PC (Python)            Edgi-Talk 板    │---

│    USB 串口采集            TensorFlow 训练         实时 AI 推理    │

│    LVGL 波形显示           INT8 量化 + C 代码       LVGL 结果展示  │## 目录结构

└──────────────────────────────────────────────────────────────────┘

``````

Edgi-Talk-IMU-AI/

## 🎯 项目特色│

├── README.md                          # 本文件

| 特性 | 描述 |├── LICENSE                            # 开源许可证

|------|------|│

| 🧠 **纯 C 推理引擎** | 不依赖 TFLite Micro / C++，纯 C 实现 INT8 量化 CNN 推理，适合资源受限 MCU |├── firmware/                          # 嵌入式固件（RT-Thread 工程）

| 📊 **LVGL 图形界面** | 数据采集阶段展示实时波形，推理阶段展示动作识别统计与趋势图 |│   ├── data-collector/                # ① 数据采集固件

| 🔄 **端到端流水线** | 从数据采集 → 训练 → 量化 → C 代码生成 → 嵌入式部署，全流程一键覆盖 |│   │   ├── applications/              #    应用层代码

| ⚡ **50Hz 实时推理** | 滑动窗口（100 采样点 / 2 秒窗口），20ms 推理周期 |│   │   │   ├── main.c                 #    主程序 + LVGL UI（波形显示）

| 📦 **超轻量模型** | 3 层 Conv2D + Dense，INT8 模型仅 ~50KB，测试准确率 **96.6%** |│   │   │   ├── imu_task.c             #    IMU 数据采集 + USB 串口发送

| 🛠️ **自动化部署** | 训练完成后一键生成 C 代码并部署到固件工程 |│   │   │   ├── cdc_acm.c              #    USB CDC 虚拟串口驱动

│   │   │   └── SConscript

## 📊 模型性能│   │   ├── board/                     #    板级支持

│   │   ├── libraries/                 #    SDK 库

| 指标 | 数值 |│   │   ├── libs/                      #    第三方库

|:----:|:----:|│   │   ├── packages/                  #    RT-Thread 软件包

| 模型架构 | 3× Conv2D + BatchNorm + MaxPool + Dense |│   │   ├── rt-thread/                 #    RT-Thread 内核

| Float32 模型大小 | 182.7 KB |│   │   ├── SConstruct                 #    SCons 构建入口

| **INT8 量化模型大小** | **50.6 KB** |│   │   ├── rtconfig.h                 #    RT-Thread 配置

| Float32 测试准确率 | 96.64% |│   │   ├── rtconfig.py                #    工具链配置

| **INT8 测试准确率** | **96.64%** |│   │   └── Kconfig                    #    Menuconfig 配置

| 推理窗口 | 2 秒（100 采样点 @ 50Hz） |│   │

| 窗口步进 | 1 秒（50 采样点，50% 重叠） |│   └── ai-inference/                  # ③ AI 推理固件

| 训练数据 | 2973 个窗口（2 人采集） |│       ├── applications/              #    应用层代码

│       │   ├── main.c                 #    主程序 + LVGL UI（识别结果展示）

<details>│       │   ├── imu_task.c             #    IMU 数据采集（6轴）

<summary>📐 模型架构详情（点击展开）</summary>│       │   ├── ai_task.c              #    AI 推理线程

│       │   ├── nn_model.c             #    神经网络模型（自动生成）

```│       │   ├── nn_model.h             #    模型头文件

Input: [100, 6, 1]  (100 timesteps × 6 IMU channels × 1)│       │   ├── preprocessing.c        #    预处理（归一化+量化）

         ││       │   ├── preprocessing.h        #    预处理头文件

    Conv2D(8, 5×1) + BatchNorm + ReLU + MaxPool(2×1)│       │   ├── ui_record.h            #    UI 录制状态结构

         ││       │   └── SConscript

    Conv2D(16, 5×1) + BatchNorm + ReLU + MaxPool(2×1)│       ├── board/                     #    板级支持

         ││       ├── libraries/                 #    SDK 库

    Conv2D(24, 3×3) + BatchNorm + ReLU + MaxPool(2×1)│       ├── libs/                      #    第三方库

         ││       ├── packages/                  #    RT-Thread 软件包

    Flatten → Dense(24) + ReLU│       ├── rt-thread/                 #    RT-Thread 内核

         ││       ├── SConstruct                 #    SCons 构建入口

    Dense(3) + Softmax│       ├── rtconfig.h                 #    RT-Thread 配置

         ││       ├── rtconfig.py                #    工具链配置

Output: [stationary, shaking, circle]│       └── Kconfig                    #    Menuconfig 配置

```│

├── model-training/                    # ② 模型训练（Python 独立环境）

</details>│   ├── requirements.txt               #    Python 依赖

│   ├── run_training.py                #    一键训练脚本

---│   ├── train_model.ipynb              #    Jupyter Notebook 交互训练

│   ├── generate_c_inference.py        #    生成纯 C 推理引擎

## 📁 项目结构│   ├── extract_model_weights.py       #    提取模型权重

│   ├── verify_c_inference.py          #    验证 C 推理正确性

```│   ├── metadata.json                  #    模型元数据

Edgi-Talk-IMU-AI/│   ├── MovementTypeDetection.improj   #    DEEPCRAFT Studio 项目文件

││   ├── Data/                          #    训练数据集

├── 📄 README.md                       # 本文件│   │   ├── Person1/                   #    采集者1的数据

├── 📄 LICENSE                         # MIT 开源许可证│   │   └── Person2/                   #    采集者2的数据

││   ├── Models/                        #    预训练模型

├── 🔧 firmware/                       # 嵌入式固件（RT-Thread 工程）│   │   └── conv2d-medium-balanced-2/

│   ├── data-collector/                # ① 数据采集固件│   ├── output/                        #    训练输出

│   │   └── applications/│   │   ├── best_model.keras           #    最佳 Keras 模型

│   │       ├── main.c                 #    主程序 + LVGL UI（实时波形）│   │   ├── model_float32.tflite       #    Float32 TFLite 模型

│   │       ├── imu_task.c             #    IMU 采集 + USB 串口发送│   │   ├── model_int8.tflite          #    INT8 量化 TFLite 模型

│   │       └── cdc_acm.c             #    USB CDC 虚拟串口驱动│   │   ├── nn_model.c                 #    生成的 C 推理代码

│   ││   │   ├── nn_model.h                 #    生成的 C 头文件

│   └── ai-inference/                  # ③ AI 推理固件│   │   ├── preprocessing.c            #    生成的预处理代码

│       └── applications/│   │   ├── preprocessing.h            #    生成的预处理头文件

│           ├── main.c                 #    主程序 + LVGL UI（识别结果）│   │   ├── normalization_params.json  #    归一化参数

│           ├── imu_task.c             #    IMU 六轴数据采集│   │   └── training_report.json       #    训练报告

│           ├── ai_task.c              #    AI 推理线程│   └── Resources/                     #    文档资源

│           ├── nn_model.c/h           #    神经网络（自动生成）│       └── imgs/

│           └── preprocessing.c/h      #    预处理（自动生成）│

│├── tools/                             # 辅助工具

├── 🤖 model-training/                 # ② 模型训练（Python）│   ├── record.py                      #    PC 端串口数据录入脚本

│   ├── run_training.py                #    一键训练脚本│   └── deploy_model.py                #    一键部署脚本（训练输出→固件）

│   ├── train_model.ipynb              #    Jupyter Notebook 交互训练│

│   ├── generate_c_inference.py        #    生成纯 C 推理引擎└── docs/                              # 项目文档

│   ├── extract_model_weights.py       #    提取模型权重    ├── hardware.md                    #    硬件接口说明

│   ├── verify_c_inference.py          #    验证 C 推理正确性    └── figures/                       #    硬件接口图片

│   ├── requirements.txt               #    Python 依赖        ├── 1.png ~ 7.png

│   ├── Data/                          #    训练数据集（2人采集）        └── ...

│   ├── output/                        #    训练输出（模型 + C 代码）```

│   └── Models/                        #    预训练模型

│---

├── 🛠️ tools/                          # 辅助工具

│   ├── record.py                      #    PC 端串口数据录入## 快速开始

│   └── deploy_model.py                #    一键部署（训练输出 → 固件）

│### 环境要求

└── 📖 docs/                           # 项目文档

    ├── hardware.md                    #    硬件接口说明| 组件 | 要求 |

    └── figures/                       #    硬件接口图|------|------|

```| **开发板** | Edgi-Talk (PSoC Edge, CY8CKIT-062S2-AI 或兼容板) |

| **嵌入式 IDE** | RT-Thread Studio 2.x+ |

> **注意**：固件工程的 `rt-thread/`、`libraries/`、`packages/` 等第三方依赖未包含在仓库中，请参照[环境搭建](#环境搭建)章节获取。| **交叉编译器** | arm-none-eabi-gcc |

| **Python** | 3.8+ (用于模型训练) |

---| **Python 依赖** | TensorFlow 2.12~2.16, NumPy, Pandas, scikit-learn |



## 🚀 快速开始### 第一步：数据采集



### 环境要求1. 使用 RT-Thread Studio 打开 `firmware/data-collector/` 工程

2. 编译并烧录至 Edgi-Talk 开发板

| 组件 | 要求 |3. 用 USB 数据线连接开发板至 PC

|------|------|4. 运行数据录入脚本：

| 🖥️ **开发板** | Edgi-Talk (Infineon PSoC Edge, M55 应用核) |

| 🔨 **嵌入式 IDE** | RT-Thread Studio 2.x+ |```bash

| ⚙️ **交叉编译器** | arm-none-eabi-gcc |cd tools/

| 🐍 **Python** | 3.8+（用于模型训练） |# 修改 record.py 中的 COM_PORT 为实际串口号

| 📦 **Python 依赖** | TensorFlow 2.12~2.16, NumPy, Pandas, scikit-learn |python record.py

```

### 环境搭建

5. 在开发板 LCD 上点击 **Start Recording** 按钮开始录制

#### 固件开发环境6. 执行目标动作（静止 / 摇晃 / 画圈），完成后按 Ctrl+C 结束

7. 数据保存为 CSV 文件

1. 安装 [RT-Thread Studio](https://www.rt-thread.org/studio.html)

2. 安装 arm-none-eabi-gcc 工具链（RT-Thread Studio 内置或手动安装）### 第二步：模型训练

3. 通过 RT-Thread Studio 的包管理器获取以下依赖：

   - **RT-Thread 内核**（v5.x）```bash

   - **LVGL 软件包**（v9.x）cd model-training/

   - **CMSIS 5**（DSP / NN 库）

4. 获取 Infineon PSoC Edge SDK 库文件，放置到 `firmware/*/libraries/` 目录# 创建虚拟环境（推荐）

python -m venv venv

#### Python 训练环境venv\Scripts\activate      # Windows

# source venv/bin/activate  # Linux/macOS

```bash

cd model-training/# 安装依赖

python -m venv venvpip install -r requirements.txt

# Windows:

venv\Scripts\activate# 一键训练（包含数据加载、训练、量化、C代码生成）

# Linux/macOS:python run_training.py

# source venv/bin/activate```



pip install -r requirements.txt或使用 Jupyter Notebook 交互式训练：

```

```bash

### 第一步：数据采集jupyter notebook train_model.ipynb

```

```bash

# 1. 用 RT-Thread Studio 打开 firmware/data-collector/ 工程训练完成后，`output/` 目录下会生成：

# 2. 编译并烧录至 Edgi-Talk 开发板- `model_int8.tflite` — INT8 量化模型

# 3. USB 连接开发板至 PC- `nn_model.c` / `nn_model.h` — 纯 C 推理引擎

- `preprocessing.c` / `preprocessing.h` — 预处理代码

cd tools/- `training_report.json` — 训练报告（准确率约 96.6%）

# 修改 record.py 中的 COM_PORT 为实际串口号

python record.py### 第三步：部署推理



# 4. 在开发板 LCD 上点击 [Start Recording] 开始录制**方式一：自动部署**

# 5. 执行目标动作（静止 / 摇晃 / 画圈）

# 6. 按 Ctrl+C 结束，数据保存为 CSV 文件```bash

```cd tools/

python deploy_model.py

### 第二步：模型训练```



```bash**方式二：手动复制**

cd model-training/

将 `model-training/output/` 下生成的 C 文件复制到固件工程中：

# 一键训练（数据加载 → 训练 → 量化 → C 代码生成）

python run_training.py```

```nn_model.c      → firmware/ai-inference/applications/nn_model.c

nn_model.h      → firmware/ai-inference/applications/nn_model.h

<details>preprocessing.c → firmware/ai-inference/applications/preprocessing.c

<summary>或使用 Jupyter Notebook 交互式训练</summary>preprocessing.h → firmware/ai-inference/applications/preprocessing.h

```

```bash

jupyter notebook train_model.ipynb然后：

```1. 使用 RT-Thread Studio 打开 `firmware/ai-inference/` 工程

2. 编译并烧录至 Edgi-Talk 开发板

</details>3. 开发板上电后 LCD 显示动作识别界面，点击 **REC** 开始实时识别



训练完成后，`output/` 目录下生成：---



| 文件 | 说明 |## 模型架构

|------|------|

| `model_int8.tflite` | INT8 量化 TFLite 模型 |```

| `nn_model.c` / `nn_model.h` | 纯 C 推理引擎代码 |Input: [100, 6, 1]  (100 timesteps × 6 IMU channels × 1)

| `preprocessing.c` / `preprocessing.h` | 预处理代码（归一化参数） |         │

| `training_report.json` | 训练报告 |    Conv2D(8, 5×1) + BatchNorm + ReLU + MaxPool(2×1)

         │

### 第三步：部署推理    Conv2D(16, 5×1) + BatchNorm + ReLU + MaxPool(2×1)

         │

**方式一：自动部署（推荐）**    Conv2D(24, 3×3) + BatchNorm + ReLU + MaxPool(2×1)

         │

```bash    Flatten(1728)

cd tools/         │

python deploy_model.py    Dense(24) + ReLU

```         │

    Dense(3) + Softmax

**方式二：手动复制**         │

Output: [stationary, shaking, circle]

``````

model-training/output/nn_model.c      → firmware/ai-inference/applications/

model-training/output/nn_model.h      → firmware/ai-inference/applications/| 指标 | 数值 |

model-training/output/preprocessing.c → firmware/ai-inference/applications/|------|------|

model-training/output/preprocessing.h → firmware/ai-inference/applications/| Float32 模型大小 | 182.7 KB |

```| INT8 量化模型大小 | 50.6 KB |

| 测试集准确率 (Float32) | 96.64% |

然后用 RT-Thread Studio 打开 `firmware/ai-inference/` 工程，编译烧录即可。| 测试集准确率 (INT8) | 96.64% |

| 推理窗口 | 2 秒（100 采样点 @ 50Hz） |

上电后 LCD 显示动作识别界面，点击 **REC** 按钮开始实时识别 🎉| 窗口步进 | 1 秒（50 采样点） |



------



## 🔧 硬件平台## 硬件平台



本项目基于 **Edgi-Talk 开发板**，核心芯片为 Infineon PSoC Edge 系列。本项目基于 **Edgi-Talk 开发板**，核心芯片为 PSoC Edge 系列，使用 **M55 应用核**（Cortex-M55）。



| 组件 | 规格 |- **IMU 传感器**：LSM6DS3（6轴加速度计 + 陀螺仪），通过软件 I2C 驱动（GPIO P8.0/P8.1）

|------|------|- **显示屏**：MIPI 接口 LCD（512×800 分辨率）

| **MCU** | PSoC Edge M55（Cortex-M55 应用核） |- **USB**：USB HS CDC 虚拟串口，用于数据采集阶段的数据传输

| **IMU** | ST LSM6DS3（6 轴加速度计 + 陀螺仪） |

| **IMU 接口** | 软件 I2C（SCL: P8.0, SDA: P8.1），地址 0x6A |详细硬件接口请参考 [docs/hardware.md](docs/hardware.md)。

| **显示屏** | MIPI DSI LCD，512 × 800 分辨率 |

| **USB** | USB 2.0 HS，CDC ACM 虚拟串口（CherryUSB） |---

| **调试器** | 板载 DAP |

## 技术栈

> 📖 详细硬件接口说明请参考 [docs/hardware.md](docs/hardware.md)

| 层级 | 技术 |

---|------|------|

| RTOS | RT-Thread 5.x |

## 🧰 技术栈| GUI | LVGL 9.x |

| AI 框架（训练） | TensorFlow / Keras |

| 层级 | 技术 | 说明 || AI 推理（嵌入式） | 纯 C INT8 量化推理引擎（无外部依赖） |

|------|------|------|| USB | CherryUSB (CDC ACM) |

| **RTOS** | RT-Thread 5.x | 实时操作系统内核 || IMU 驱动 | 软件 I2C + LSM6DS3 |

| **GUI** | LVGL 9.x | 嵌入式图形界面 || 构建系统 | SCons |

| **AI 训练** | TensorFlow / Keras | CNN 模型训练与量化 || 交叉编译 | arm-none-eabi-gcc |

| **AI 推理** | 纯 C INT8 引擎 | 零外部依赖，自动代码生成 |

| **USB** | CherryUSB (CDC ACM) | 数据采集串口通信 |---

| **IMU** | 软件 I2C + LSM6DS3 | 六轴运动传感器 |

| **构建** | SCons | RT-Thread 标准构建系统 |## Introduction

| **编译** | arm-none-eabi-gcc | ARM Cortex-M 交叉编译 |

This project is a complete **end-to-end embedded AI motion recognition system** built on the **Edgi-Talk development board** (PSoC Edge M55 core) with **RT-Thread RTOS**. It uses the onboard IMU (LSM6DS3, 6-axis accelerometer + gyroscope) to recognize three motion types in real time: **Stationary**, **Shaking**, and **Circle**.

---

### Key Features

## ✨ Introduction

- **Pure C Inference Engine**: No TFLite Micro or C++ required — standalone INT8 quantized CNN inference in pure C

This project is a complete **end-to-end embedded AI motion recognition system** built on the **Edgi-Talk development board** (Infineon PSoC Edge M55 core) with **RT-Thread RTOS**. It uses the onboard IMU (LSM6DS3, 6-axis accelerometer + gyroscope) to recognize three motion types in real time:- **LVGL GUI**: Real-time waveform display during data collection; action recognition statistics during inference

- **End-to-End Pipeline**: Data collection → Model training → INT8 quantization → C code generation → Embedded deployment

| Motion | Description |- **50Hz Real-time Inference**: Sliding window (100 samples / 2s window) for real-time classification

|:------:|-------------|- **Lightweight CNN**: 3× Conv2D + Dense, INT8 model ~50KB, 96.6% test accuracy

| **Stationary** | Still (on table / held steady) |

| **Shaking** | Shaking motion |### Workflow

| **Circle** | Circular motion |

1. **Data Collection** (`firmware/data-collector/`): Flash to Edgi-Talk board, record IMU data via USB CDC to PC

### Key Features2. **Model Training** (`model-training/`): Train CNN with TensorFlow, quantize to INT8, generate pure C inference code

3. **AI Deployment** (`firmware/ai-inference/`): Flash to Edgi-Talk board, real-time motion recognition with LVGL UI

- 🧠 **Pure C Inference Engine** — No TFLite Micro or C++ dependency. Standalone INT8 quantized CNN in pure C

- 📊 **LVGL GUI** — Real-time waveform during data collection; recognition stats during inferenceSee the Chinese section above for detailed setup instructions.

- 🔄 **End-to-End Pipeline** — Data collection → Training → INT8 quantization → C code generation → Embedded deployment

- ⚡ **50Hz Real-time Inference** — Sliding window (100 samples / 2s window) with 20ms inference cycle---

- 📦 **Ultra-lightweight** — 3× Conv2D + Dense, INT8 model ~50KB, **96.6% test accuracy**

## 开源许可

### Quick Start (English)

本项目采用 [MIT License](LICENSE) 开源。

1. **Collect Data**: Flash `firmware/data-collector/` → record IMU data via USB CDC

2. **Train Model**: `cd model-training/ && python run_training.py`## 致谢

3. **Deploy**: `cd tools/ && python deploy_model.py`

4. **Run**: Flash `firmware/ai-inference/` → real-time motion recognition on LCD- [RT-Thread](https://www.rt-thread.org/) — 实时操作系统

- [LVGL](https://lvgl.io/) — 嵌入式图形库

> See the Chinese sections above for detailed step-by-step instructions.- [TensorFlow](https://www.tensorflow.org/) — 机器学习框架

- [CherryUSB](https://github.com/cherry-embedded/CherryUSB) — USB 协议栈

---- [Infineon DEEPCRAFT Studio](https://softwaretools.infineon.com/) — 原始数据集来源



## 📜 开源许可---



本项目采用 [MIT License](LICENSE) 开源许可。## 贡献



## 🙏 致谢欢迎提交 Issue 和 Pull Request！



| 项目 | 说明 |1. Fork 本仓库

|------|------|2. 创建你的特性分支 (`git checkout -b feature/amazing-feature`)

| [RT-Thread](https://www.rt-thread.org/) | 实时操作系统 |3. 提交你的更改 (`git commit -m 'Add amazing feature'`)

| [LVGL](https://lvgl.io/) | 嵌入式图形库 |4. 推送到分支 (`git push origin feature/amazing-feature`)

| [TensorFlow](https://www.tensorflow.org/) | 机器学习框架 |5. 打开 Pull Request

| [CherryUSB](https://github.com/cherry-embedded/CherryUSB) | USB 协议栈 |
| [Infineon DEEPCRAFT Studio](https://softwaretools.infineon.com/) | 原始数据集来源 |

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'Add amazing feature'`
4. 推送分支：`git push origin feature/amazing-feature`
5. 发起 Pull Request

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请点个 Star！⭐**

</div>
