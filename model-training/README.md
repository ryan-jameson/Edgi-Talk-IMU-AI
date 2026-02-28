# 模型训练 — Movement Type Detection

## 概述

本目录包含 Edgi-Talk IMU AI 项目的完整 **模型训练流水线**，基于 TensorFlow/Keras 实现 CNN 动作分类模型的训练、INT8 量化及纯 C 推理代码生成。

## 文件说明

| 文件 | 说明 |
|------|------|
| `run_training.py` | 一键训练脚本（全流程自动化） |
| `train_model.ipynb` | Jupyter Notebook 交互式训练 |
| `generate_c_inference.py` | 将 TFLite 模型转换为纯 C 推理引擎 |
| `extract_model_weights.py` | 从 Keras 模型提取权重 |
| `verify_c_inference.py` | 对比验证 C 推理与 TFLite 推理一致性 |
| `requirements.txt` | Python 依赖列表 |
| `metadata.json` | 模型元数据（分类类型、传感器等） |
| `MovementTypeDetection.improj` | DEEPCRAFT Studio 项目文件（可选） |

## 目录结构

```
model-training/
├── Data/                    # 训练数据集
│   ├── Person1/             # 采集者1（19个session）
│   └── Person2/             # 采集者2
├── Models/                  # 预训练模型参考
├── output/                  # 训练输出
│   ├── model_int8.tflite    # INT8 量化模型（~50KB）
│   ├── nn_model.c/h         # 自动生成的 C 推理代码
│   ├── preprocessing.c/h    # 自动生成的预处理代码
│   └── training_report.json # 训练报告
└── Resources/               # 文档资源图片
```

## 快速使用

```bash
# 安装依赖
pip install -r requirements.txt

# 一键训练（包含全部 10 个步骤）
python run_training.py
```

### 训练流水线（10 步）

1. **数据加载** — 自动发现 `Data/` 下所有 session
2. **滑动窗口** — 100 帧窗口（2s @ 50Hz），50% 重叠
3. **数据归一化** — 逐通道 Min-Max 归一化
4. **数据划分** — 80% 训练 / 10% 验证 / 10% 测试，类别平衡
5. **构建模型** — 3 层 Conv2D + BatchNorm + MaxPool + Dense
6. **训练** — 150 epochs，类别权重平衡，EarlyStopping
7. **评估** — 测试集准确率 + 混淆矩阵
8. **INT8 量化** — TFLite 全整数量化
9. **C 代码生成** — 生成纯 C 推理引擎（nn_model.c/h + preprocessing.c/h）
10. **验证** — 对比验证 C 推理与 TFLite 结果一致性

### 训练结果

| 指标 | 数值 |
|------|------|
| Float32 测试准确率 | 96.64% |
| INT8 测试准确率 | 96.64% |
| INT8 模型大小 | 50.6 KB |
| 训练数据量 | 2973 个窗口 |

## 数据格式

训练数据来自 DEEPCRAFT Studio 采集，每个 session 包含：
- `IMU-Data.data` — 六轴 IMU 数据（时间戳 + 3 轴加速度 + 3 轴陀螺仪）
- `Live-Labeling.label` — 时间区间标注（shaking / circle，未标注区域为 stationary）

## 添加新数据

1. 使用 `firmware/data-collector` 固件 + `tools/record.py` 采集新数据
2. 按照 `Data/PersonX/SessionName/` 格式组织数据
3. 重新运行 `python run_training.py`
4. 使用 `tools/deploy_model.py` 部署到固件
