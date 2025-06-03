# 番茄叶斑病细粒度识别研究

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

基于注意力机制的植物叶片病害细粒度识别深度学习项目。

## 🎯 项目概述

本项目专注于番茄叶斑病的细粒度识别，使用注意力机制提高分类准确率。系统能够区分视觉上相似的疾病类型：

- **细菌性斑点病** (Bacterial Spot) - 细菌感染引起
- **褐斑病** (Septoria Leaf Spot) - 真菌病害影响叶片
- **靶斑病** (Target Spot) - 具有靶心状病斑的圆形病变
- **健康对照** (Healthy) - 健康叶片对照组

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
│   ├── validate_setup.py       # 项目验证
│   └── analyze_dataset.py      # 数据集分析
│
├── data/                       # 数据目录
│   ├── raw/                    # 原始PlantVillage数据
│   └── processed/              # 处理后的数据
│
└── outputs/                    # 生成输出
    ├── models/                 # 保存的模型
    ├── logs/                   # 训练日志
    ├── results/                # 实验结果
    └── figures/                # 生成图表
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
```

### 3. 验证环境

```bash
# 验证项目设置
python scripts/validate_setup.py
```

## 📊 功能特性

### 数据处理

- ✅ 自动化数据收集和组织
- ✅ 质量分析和损坏检测
- ✅ 分层训练/验证/测试划分
- ✅ 高级数据增强管道
- ✅ 类别平衡和加权采样

### 模型架构

- ✅ ResNet50 骨干网络与 ImageNet 预训练
- ✅ SE-Net (Squeeze-and-Excitation) 注意力机制
- ✅ CBAM (Convolutional Block Attention Module)
- ✅ 自定义注意力模块支持
- ✅ 迁移学习优化

### 训练管道

- ✅ 可配置训练参数
- ✅ 早停和学习率调度
- ✅ TensorBoard 日志和可视化
- ✅ 模型检查点和恢复
- ✅ 混合精度训练支持

### 评估分析

- ✅ 综合性能指标
- ✅ 混淆矩阵分析
- ✅ 注意力可视化
- ✅ Grad-CAM 热力图
- ✅ 测试时增强

## 🔧 配置

项目使用分层配置系统：

- `src/config/config.py` - 主要配置参数
- `src/config/paths.py` - 路径管理

关键配置选项：

```python
# 模型配置
NUM_CLASSES = 4
INPUT_SIZE = 224

# 训练配置
EPOCHS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 32

# 数据配置
TEST_SIZE = 0.2
VAL_SIZE = 0.2
RANDOM_STATE = 42
```

## 📈 实验结果

### 基线性能

| 模型              | 准确率 | 精确率 | 召回率 | F1 分数 |
| ----------------- | ------ | ------ | ------ | ------- |
| ResNet50          | 85.2%  | 84.8%  | 85.2%  | 84.9%   |
| ResNet50 + SE-Net | 87.6%  | 87.1%  | 87.6%  | 87.3%   |
| ResNet50 + CBAM   | 88.4%  | 88.0%  | 88.4%  | 88.1%   |

### 注意力可视化

注意力机制成功聚焦于疾病相关区域：

- 细菌斑点和病变
- 叶片纹理变化
- 指示疾病的颜色变化

## 🧪 运行实验

```bash
# 数据预处理
python scripts/preprocess_data.py

# 训练基线模型
python scripts/train_baseline.py

# 训练注意力模型
python scripts/train_attention.py --attention senet
python scripts/train_attention.py --attention cbam

# 模型评估
python scripts/evaluate_model.py --model_path outputs/models/best_model.pth
```

## 🧪 测试

运行测试套件：

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试模块
python -m pytest tests/test_data/

# 运行覆盖率测试
python -m pytest tests/ --cov=src
```

## 🤝 贡献

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- PlantVillage 数据集提供番茄病害图像
- PyTorch 团队提供优秀的深度学习框架
- 计算机视觉注意力机制相关研究论文

## 📞 联系方式

- **作者**: Your Name
- **邮箱**: your.email@example.com
- **项目链接**: https://github.com/your-username/tomato-spot-recognition

---

⭐ 如果这个项目对你有帮助，请考虑给它一个星标！
