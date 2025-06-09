# API 参考文档

## 概述

本文档提供植物叶片病害识别系统的详细 API 参考。

## 核心模块

### 1. 模型模块 (`src.models`)

#### ResNetSE

SE-Net (Squeeze-and-Excitation Networks) 实现

```python
from src.models.attention_models import ResNetSE

model = ResNetSE(
    num_classes=10,          # 类别数量
    reduction=16,            # SE模块的降维率
    dropout_rate=0.7         # Dropout率
)
```

**参数:**

-   `num_classes` (int): 分类类别数
-   `reduction` (int): SE 模块通道降维率，默认 16
-   `dropout_rate` (float): Dropout 比率，默认 0.5

**方法:**

-   `forward(x)`: 前向传播
-   `get_attention_weights()`: 获取注意力权重

#### ResNetCBAM

CBAM (Convolutional Block Attention Module) 实现

```python
from src.models.attention_models import ResNetCBAM

model = ResNetCBAM(
    num_classes=10,
    reduction=16,
    dropout_rate=0.7
)
```

### 2. 数据模块 (`src.data`)

#### TomatoSpotDataset

自定义数据集类

```python
from src.data.dataset import TomatoSpotDataset

dataset = TomatoSpotDataset(
    df=dataframe,           # 包含图像路径和标签的DataFrame
    transform=transform,    # torchvision变换
    label_to_idx=mapping   # 标签到索引的映射
)
```

**方法:**

-   `__len__()`: 返回数据集大小
-   `__getitem__(idx)`: 获取指定索引的样本

#### 数据预处理函数

```python
from src.data.preprocessing import preprocess_data

# 数据分割
train_df, val_df, test_df = split_data(
    df=data,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_state=42
)

# 数据增强
transform = get_train_transform(
    input_size=224,
    rotation_degrees=15,
    brightness=0.2
)
```

### 3. 训练模块 (`src.training`)

#### Trainer 类

统一的模型训练器

```python
from src.training.trainer import Trainer

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config
)

# 开始训练
trainer.train()
```

**方法:**

-   `train()`: 执行完整训练流程
-   `train_epoch()`: 训练一个 epoch
-   `validate()`: 模型验证
-   `save_checkpoint()`: 保存检查点

### 4. 评估模块 (`src.evaluation`)

#### ModelEvaluator

模型评估器

```python
from src.evaluation.evaluator import ModelEvaluator

evaluator = ModelEvaluator(model, test_loader, class_names)

# 评估模型
results = evaluator.evaluate()
```

**返回结果:**

```python
{
    'accuracy': 0.95,
    'precision': [0.94, 0.96, ...],
    'recall': [0.93, 0.97, ...],
    'f1_score': [0.935, 0.965, ...],
    'confusion_matrix': [[...], [...], ...]
}
```

### 5. 可视化模块 (`src.visualization`)

#### AttentionVisualizer

注意力可视化器

```python
from src.visualization.attention_visualizer import AttentionVisualizer

visualizer = AttentionVisualizer(model, device='cuda')
visualizer.visualize_sample(image, label, image_path, output_dir)
```

#### GradCAM

类激活映射可视化

```python
from src.visualization.grad_cam import GradCAM

grad_cam = GradCAM(model, target_layer='layer4')
heatmap = grad_cam.generate_cam(image, target_class)
```

## Web 应用 API

### Flask 路由

#### 主页

```
GET /
```

返回主页 HTML

#### 模型列表

```
GET /api/models
```

**响应:**

```json
[
    {
        "name": "SE-Net",
        "type": "attention",
        "size": "25.6 MB",
        "accuracy": "95.2%"
    }
]
```

#### 预测 API

```
POST /api/predict
```

**请求:**

-   Content-Type: multipart/form-data
-   文件字段: `image`

**响应:**

```json
{
    "prediction": "健康叶片",
    "confidence": 0.952,
    "probabilities": {
        "健康叶片": 0.952,
        "斑点病": 0.031,
        "其他": 0.017
    }
}
```

#### 训练 API

```
POST /api/train
```

**请求体:**

```json
{
    "model_type": "se_net",
    "epochs": 50,
    "learning_rate": 0.001,
    "batch_size": 32
}
```

**响应:**

```json
{
    "training_id": "uuid-string",
    "status": "started",
    "message": "训练已开始"
}
```

#### 训练状态

```
GET /api/training_status/<training_id>
```

**响应:**

```json
{
    "status": "running",
    "progress": 45.2,
    "current_epoch": 23,
    "total_epochs": 50,
    "train_acc": 0.89,
    "val_acc": 0.87,
    "train_loss": 0.234,
    "val_loss": 0.289
}
```

## 配置系统

### Config 类

```python
from src.config.config import Config

config = Config()
print(config.NUM_CLASSES)  # 10
print(config.INPUT_SIZE)   # 224
```

### YAML 配置加载

```python
from src.config.yaml_config import load_config

config = load_config('configs/training_config.yaml')
```

## 工具函数

### 性能计算

```python
from src.utils.metrics import calculate_metrics

metrics = calculate_metrics(y_true, y_pred, class_names)
```

### 模型保存/加载

```python
from src.utils.model_utils import save_model, load_model

# 保存模型
save_model(model, 'outputs/models/model.pth', metadata)

# 加载模型
model = load_model('outputs/models/model.pth', model_class)
```

### 日志工具

```python
from src.utils.logger import setup_logger

logger = setup_logger('training', 'outputs/logs/train.log')
logger.info('训练开始')
```

## 错误处理

### 自定义异常

```python
from src.utils.exceptions import ModelLoadError, DataLoadError

try:
    model = load_model(path)
except ModelLoadError as e:
    logger.error(f"模型加载失败: {e}")
```

## 示例用法

### 完整训练流程

```python
from src.config.config import Config
from src.data.dataset import create_data_loaders
from src.models.attention_models import ResNetSE
from src.training.trainer import Trainer

# 1. 加载配置
config = Config()

# 2. 准备数据
train_loader, val_loader, test_loader = create_data_loaders(config)

# 3. 创建模型
model = ResNetSE(num_classes=config.NUM_CLASSES)

# 4. 训练模型
trainer = Trainer(model, train_loader, val_loader, config)
trainer.train()

# 5. 评估模型
evaluator = ModelEvaluator(model, test_loader, config.CLASS_NAMES)
results = evaluator.evaluate()
```

### Web 应用使用

```python
# 启动Web应用
from webapp.app import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

## 依赖要求

### 核心依赖

-   torch >= 1.9.0
-   torchvision >= 0.10.0
-   numpy >= 1.21.0
-   pandas >= 1.3.0
-   PIL >= 8.3.0

### Web 应用依赖

-   Flask >= 2.3.0
-   Werkzeug >= 2.3.0

### 可视化依赖

-   matplotlib >= 3.4.0
-   seaborn >= 0.11.0

完整依赖列表请参考 `requirements.txt`。
