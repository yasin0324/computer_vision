# 基于注意力机制的植物叶片病害细粒度识别研究

> 使用深度学习和注意力机制进行番茄叶斑病细粒度识别的研究项目

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 项目概述

本项目专注于番茄叶斑病的细粒度识别，使用 PlantVillage 数据集中的四个关键类别：

-   **细菌性斑点病** (Bacterial Spot)
-   **褐斑病** (Septoria Leaf Spot)
-   **靶斑病** (Target Spot)
-   **健康对照** (Healthy)

通过集成注意力机制（SE-Net、CBAM 等）到 ResNet50 骨干网络，实现高精度的植物病害识别。

## 🏗️ 项目结构

```
tomato-spot-recognition/
├── README.md                    # 项目文档
├── requirements.txt             # Python依赖
├── .gitignore                  # Git忽略规则
│
├── src/                        # 源代码
│   ├── config/                 # 配置管理
│   ├── data/                   # 数据处理模块
│   ├── models/                 # 模型定义
│   ├── training/               # 训练管道
│   ├── evaluation/             # 评估工具
│   └── utils/                  # 工具函数
│
├── scripts/                    # 可执行脚本
│   ├── preprocess_data.py      # 数据预处理
│   ├── validate_preprocessing.py # 数据验证
│   ├── train_baseline.py       # 基线模型训练
│   ├── quick_train.py          # 快速训练测试
│   └── test_model.py           # 模型测试
│
├── data/                       # 数据目录
│   ├── raw/                    # 原始PlantVillage数据
│   └── processed/              # 处理后的数据
│
├── outputs/                    # 生成输出
│   ├── models/                 # 保存的模型
│   ├── logs/                   # 训练日志
│   ├── results/                # 实验结果
│   └── figures/                # 生成图表
│
└── docs/                       # 文档
    └── TRAINING_GUIDE.md       # 训练指南
```

## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆项目
git clone https://github.com/your-username/tomato-spot-recognition.git
cd tomato-spot-recognition

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

下载 PlantVillage 数据集并放置到 `data/raw/PlantVillage/` 目录：

```bash
# 运行数据预处理
python scripts/preprocess_data.py

# 验证数据预处理结果
python scripts/validate_preprocessing.py
```

### 3. 模型训练

```bash
# 测试训练系统
python scripts/quick_train.py

# 开始完整训练
python scripts/train_baseline.py
```

详细的训练指南请参考：[docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)

## 📊 功能特性

### 数据处理

-   ✅ 自动化数据收集和组织
-   ✅ 质量分析和损坏检测
-   ✅ 分层训练/验证/测试划分 (60%/20%/20%)
-   ✅ 高级数据增强管道
-   ✅ 类别平衡和加权采样

### 模型架构

-   ✅ ResNet50 骨干网络与 ImageNet 预训练
-   🚧 SE-Net (Squeeze-and-Excitation) 注意力机制
-   🚧 CBAM (Convolutional Block Attention Module)
-   🚧 自定义注意力模块支持
-   ✅ 迁移学习优化

### 训练管道

-   ✅ 可配置训练参数
-   ✅ 早停和学习率调度
-   ✅ TensorBoard 日志和可视化（可选）
-   ✅ 模型检查点和恢复
-   ✅ 混合精度训练支持
-   ✅ 实时训练监控

### 评估分析

-   🚧 综合性能指标
-   🚧 混淆矩阵分析
-   🚧 注意力可视化
-   🚧 Grad-CAM 热力图
-   🚧 测试时增强

## 🔧 配置

项目使用分层配置系统：

-   `src/config/config.py` - 主要配置参数

关键配置选项：

```python
# 模型配置
NUM_CLASSES = 4
INPUT_SIZE = 224
PRETRAINED = True

# 训练配置
EPOCHS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-4

# 数据配置
TEST_SIZE = 0.2
VAL_SIZE = 0.2
RANDOM_STATE = 42

# 早停配置
EARLY_STOPPING = {
    'patience': 15,
    'min_delta': 0.001
}
```

## 📈 实验结果

### 数据集统计

-   **总样本数**: 6,893 张图像
-   **训练集**: 4,135 张 (60%)
-   **验证集**: 1,379 张 (20%)
-   **测试集**: 1,379 张 (20%)

### 类别分布

| 类别         | 样本数 | 比例  |
| ------------ | ------ | ----- |
| 细菌性斑点病 | 2,127  | 30.9% |
| 健康对照     | 1,591  | 23.1% |
| 褐斑病       | 1,771  | 25.7% |
| 靶斑病       | 1,404  | 20.4% |

### 基线性能

| 模型              | 准确率 | 精确率 | 召回率 | F1 分数 |
| ----------------- | ------ | ------ | ------ | ------- |
| ResNet50 基线     | 🚧     | 🚧     | 🚧     | 🚧      |
| ResNet50 + SE-Net | 🚧     | 🚧     | 🚧     | 🚧      |
| ResNet50 + CBAM   | 🚧     | 🚧     | 🚧     | 🚧      |

_注：性能指标将在训练完成后更新_

## 🧪 运行实验

### 数据预处理

```bash
python scripts/preprocess_data.py
python scripts/validate_preprocessing.py
```

### 模型训练

```bash
# 快速测试训练系统
python scripts/quick_train.py

# 基线模型训练
python scripts/train_baseline.py

# 带早停的训练
python scripts/train_baseline.py --early_stopping --patience 10

# 自定义实验
python scripts/train_baseline.py \
    --epochs 50 \
    --learning_rate 0.001 \
    --experiment_name "custom_experiment"
```

### 模型评估

```bash
# 模型评估（待实现）
python scripts/evaluate_model.py --model_path outputs/models/best_model.pth
```

### 训练监控

如果安装了 TensorBoard：

```bash
# 安装TensorBoard
pip install tensorboard

# 启动TensorBoard
tensorboard --logdir outputs/logs
```

## 🧪 测试

运行测试套件：

```bash
# 测试模型创建和前向传播
python scripts/test_model.py

# 快速训练测试
python scripts/quick_train.py
```

## 📁 输出文件

训练完成后会生成以下文件：

```
outputs/
├── models/{experiment_name}/
│   ├── checkpoint_epoch_*.pth      # 定期检查点
│   ├── best_checkpoint_epoch_*.pth # 最佳模型
│   └── training_history.json      # 训练历史
├── logs/{experiment_name}/         # TensorBoard日志
└── results/
    └── {experiment_name}_results.json # 结果摘要
```

---

⭐ 如果这个项目对你有帮助，请考虑给它一个星标！
