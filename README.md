# 🌱 植物叶片病害识别系统

这是一个基于注意力机制的植物叶片病害细粒度识别项目。项目旨在解决不同病害间视觉特征高度相似的识别难题。实现方式为在 ResNet-50 骨干网络上，集成并对比 SE-Net 和 CBAM 两种注意力机制的效果，并通过 Flask Web 框架提供一个交互界面。

## ✨ 功能特性

-   **高精度识别**：采用基于注意力机制的深度学习模型（ResNet+SE, ResNet+CBAM），专注于细粒度特征，确保高准确率。
-   **Web 交互界面**：提供简洁直观的 Web 页面，用户可轻松上传图片并获取识别结果。
-   **实时预测**：上传图片后，系统能够迅速给出预测结果。
-   **可扩展性**：项目结构清晰，方便进行二次开发、模型替换或功能扩展。

## 🛠️ 技术栈

-   **后端**：Python, PyTorch, Flask
-   **数据处理**：Pandas, NumPy, OpenCV
-   **前端**：HTML, CSS, JavaScript (通过 Flask Templates)
-   **可视化**：Matplotlib, Seaborn, TensorBoard
-   **实验跟踪**：Weights & Biases (wandb)

## 📂 项目结构

```
visual/
├── .conda/            # Conda 环境配置
├── .git/              # Git 仓库
├── configs/           # 配置文件目录 (模型、训练等配置)
├── data/              # 数据集目录
├── logs/              # 日志文件
├── outputs/           # 输出目录 (训练好的模型、结果等)
├── scripts/           # 辅助脚本
├── src/               # 项目核心源码
│   ├── data/          # 数据加载和预处理
│   ├── models/        # 深度学习模型定义
│   ├── training/      # 模型训练逻辑
│   ├── evaluation/    # 模型评估逻辑
│   ├── visualization/ # 可视化代码
│   ├── config/        # 配置管理
│   └── utils/         # 工具函数
├── tests/             # 测试代码
├── webapp/            # Flask Web 应用
│   ├── static/        # 静态文件 (CSS, JS, Images)
│   ├── templates/     # HTML 模板
│   ├── app.py         # Flask 应用主文件
│   └── utils.py       # Web 应用工具函数
├── .gitignore         # Git 忽略文件
├── README.md          # 项目说明
├── requirements.txt   # Python 依赖
└── run_webapp.py      # Web 应用启动脚本
```

## Dataset 说明

本项目默认使用经典的 **PlantVillage** 数据集的一个子集，主要针对番茄的四种叶片状况：**细菌性斑点病 (Bacterial spot)**、**褐斑病 (Septoria leaf spot)**、**靶斑病 (Target Spot)** 和 **健康叶片 (Healthy)**。

为了确保代码能够正确加载数据，请按照以下结构组织数据集：

**1. 准备原始数据**

首先，请将原始数据集放入 `data/raw/` 目录下，并按类别分好文件夹。

```
data/
└── raw/
    ├── Tomato___Bacterial_spot/
    │   ├── image_001.jpg
    │   └── image_002.jpg
    ├── Tomato___Septoria_leaf_spot/
    ├── Tomato___Target_Spot/
    └── Tomato___healthy/
```

**2. 运行预处理脚本**

项目提供了一个脚本，可以自动将 `data/raw/` 下的数据集进行划分（训练集/验证集）并保存到 `data/processed/` 目录，以供模型训练使用。

```bash
python scripts/preprocess_data.py
```

该脚本会创建如下所示的目录结构：

```
data/
└── processed/
    ├── train/
    │   ├── Tomato___Bacterial_spot/
    │   └── ...
    └── val/
        ├── Tomato___Bacterial_spot/
        └── ...
```

可以在 `src/data/` 目录下的数据加载脚本中修改路径或数据加载逻辑以适应自己的数据集。

## ⚙️ 安装与设置

1.  **克隆项目**

    ```bash
    git clone https://github.com/yasin0324/computer_vision.git
    cd visual
    ```

2.  **创建并激活 Conda 环境** (推荐)

    ```bash
    conda create -n visual_env python=3.9
    conda activate visual_env
    ```

3.  **安装依赖**

    ```bash
    pip install -r requirements.txt
    ```

    _注意：请根据 CUDA 版本安装对应的 PyTorch。详细信息请参考 [PyTorch 官网](https://pytorch.org/get-started/locally/)_

4.  **下载预训练模型**
    将训练好的模型文件（例如 `best_model.pth`）放置到 `outputs/models/` 目录下。

## 🚀 使用方法

### 启动 Web 应用

在项目根目录下运行以下命令：

```bash
python run_webapp.py
```

启动成功后，将看到以下输出：

```
============================================================
🌱 植物叶片病害识别系统
============================================================
🚀 启动Web应用...
📍 访问地址: http://localhost:5000
📍 病害识别: http://localhost:5000/predict
============================================================
```

在浏览器中打开 `http://localhost:5000` 即可开始使用。

## 🏋️‍♂️ 模型训练

如果想重新训练模型：

1.  **准备数据集**：按照 "Dataset 说明" 的步骤准备并预处理数据集。
2.  **配置训练参数**：修改 `configs/` 目录下的相关配置文件，例如 `train_config.yaml`，设置学习率、批大小、训练轮次等。
3.  **开始训练**：`scripts/` 目录下提供了针对不同模型的训练脚本。选择一个来运行，例如训练基线模型：
    ```bash
    python scripts/train_baseline.py --config configs/training_config.yaml
    ```
    _可以选择 `train_se_net.py` 或 `train_cbam.py` 来训练带有注意力机制的模型。_

---
