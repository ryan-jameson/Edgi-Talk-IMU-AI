# AI 推理固件 — Edgi-Talk IMU 动作识别（详细说明）

## 简介

本工程是 Edgi-Talk IMU AI 项目的 **推理部署固件**，基于 RT-Thread 实时操作系统运行在 **M55 应用核**（Cortex-M55）上。

### 核心架构

```
┌──────────────────────────────────────┐
│            RT-Thread RTOS            │
├──────────┬───────────┬───────────────┤
│ IMU 线程  │  AI 线程   │   LVGL UI    │
│ 50Hz采集  │ CNN推理    │  结果展示     │
│ LSM6DS3  │ INT8量化   │  统计图表     │
├──────────┴───────────┴───────────────┤
│     HAL Drivers / Board BSP          │
└──────────────────────────────────────┘
```

### 文件说明

| 文件 | 说明 |
|------|------|
| `applications/main.c` | 系统入口 + LVGL 动作识别 UI（512×800 LCD） |
| `applications/imu_task.c` | IMU 采集线程，软件 I2C 驱动 LSM6DS3 |
| `applications/ai_task.c` | AI 推理线程，纯 C INT8 CNN 推理引擎 |
| `applications/nn_model.c/h` | 模型权重与推理代码（由 `generate_c_inference.py` 自动生成） |
| `applications/preprocessing.c/h` | 数据归一化与 INT8 量化（由训练脚本自动生成） |
| `applications/ui_record.h` | UI 状态与推理线程共享的数据结构 |

### 推理流程

1. **IMU 线程** 以 50Hz 读取加速度计 + 陀螺仪原始数据
2. **AI 线程** 将数据归一化、量化为 INT8，填入 100 帧滑动窗口
3. 窗口满后执行 3 层 Conv2D + Dense CNN 推理
4. 输出 3 分类概率 → 稳定判断（连续 2 帧一致）→ 更新 UI 统计

### 使用方法

1. 确保已获取 RT-Thread 内核、SDK 库和软件包（参见根目录 README）
2. 使用 RT-Thread Studio 打开本工程
3. 编译 → 通过板载 DAP 下载至开发板
4. 上电后 LCD 显示识别界面，点击 **REC** 开始实时识别

### 注意事项

- `nn_model.c/h` 和 `preprocessing.c/h` 由模型训练流程自动生成，一般无需手动修改
- 如需更新模型，在 `model-training/` 下重新训练后运行 `tools/deploy_model.py` 即可
- 如需修改硬件配置，使用 `tools/device-configurator/device-configurator.exe` 打开 `libs/TARGET_APP_KIT_PSE84_EVAL_EPC2/config/design.modus`
