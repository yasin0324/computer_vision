# 开发指南

## 环境设置

### 1. 前置要求

-   Python 3.8+
-   CUDA 11.0+ (可选，用于 GPU 加速)
-   Git

### 2. 项目设置

```bash
# 克隆项目
git clone <repository-url>
cd visual

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt

# 安装开发依赖
pip install -r requirements-dev.txt  # 如果存在
```

### 3. 数据准备

```bash
# 创建数据目录
mkdir -p data/raw data/processed

# 下载数据集（请替换为实际数据源）
# python scripts/download_data.py

# 预处理数据
python scripts/preprocess_data.py
```

## 项目结构详解

```
visual/
├── src/                    # 核心源代码
│   ├── config/            # 配置管理
│   ├── data/              # 数据处理
│   ├── models/            # 模型定义
│   ├── training/          # 训练逻辑
│   ├── evaluation/        # 评估工具
│   ├── utils/             # 工具函数
│   └── visualization/     # 可视化工具
├── scripts/               # 可执行脚本
├── webapp/                # Web应用
├── tests/                 # 测试代码
├── configs/               # 配置文件
├── docs/                  # 文档
└── outputs/               # 输出结果
```

## 开发工作流

### 1. 代码风格

项目使用以下代码风格工具：

```bash
# 代码格式化
black src/ scripts/ webapp/

# 代码检查
flake8 src/ scripts/ webapp/

# 导入排序
isort src/ scripts/ webapp/
```

### 2. Git 工作流

```bash
# 创建功能分支
git checkout -b feature/new-feature

# 提交代码
git add .
git commit -m "feat: 添加新功能"

# 推送分支
git push origin feature/new-feature

# 创建Pull Request
```

### 3. 提交信息规范

使用 Conventional Commits 规范：

-   `feat:` 新功能
-   `fix:` 修复 bug
-   `docs:` 文档更新
-   `style:` 代码格式调整
-   `refactor:` 重构代码
-   `test:` 添加测试
-   `chore:` 构建或辅助工具的变动

## 添加新功能

### 1. 添加新模型

1. 在 `src/models/` 目录创建模型文件
2. 继承基础模型类
3. 实现必要的方法
4. 添加配置支持
5. 编写测试

示例：

```python
# src/models/new_model.py
import torch.nn as nn
from .base_model import BaseModel

class NewAttentionModel(BaseModel):
    def __init__(self, num_classes, **kwargs):
        super().__init__()
        # 实现模型结构

    def forward(self, x):
        # 实现前向传播
        pass

    def get_attention_weights(self):
        # 返回注意力权重
        pass
```

### 2. 添加新的数据处理功能

1. 在 `src/data/` 目录添加处理函数
2. 更新数据加载器
3. 添加相应的配置
4. 编写测试

### 3. 添加新的评估指标

1. 在 `src/evaluation/metrics.py` 添加指标函数
2. 更新评估器类
3. 在配置中添加新指标
4. 更新可视化

## 测试

### 1. 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_models.py

# 运行覆盖率测试
pytest --cov=src
```

### 2. 编写测试

在 `tests/` 目录下创建对应的测试文件：

```python
# tests/test_new_feature.py
import unittest
import torch
from src.models.new_model import NewAttentionModel

class TestNewAttentionModel(unittest.TestCase):
    def setUp(self):
        self.model = NewAttentionModel(num_classes=10)

    def test_forward_pass(self):
        x = torch.randn(1, 3, 224, 224)
        output = self.model(x)
        self.assertEqual(output.shape, (1, 10))

    def test_attention_weights(self):
        weights = self.model.get_attention_weights()
        self.assertIsNotNone(weights)
```

## 配置管理

### 1. 添加新配置

在 `configs/training_config.yaml` 中添加配置项：

```yaml
# 新模型配置
new_model:
    architecture: "custom"
    hidden_dim: 512
    attention_heads: 8
```

### 2. 在代码中使用配置

```python
from src.config.yaml_config import load_config

config = load_config('configs/training_config.yaml')
model_config = config['models']['new_model']
```

## 调试技巧

### 1. 日志配置

```python
import logging
from src.utils.logger import setup_logger

logger = setup_logger('debug', level=logging.DEBUG)
logger.debug('调试信息')
```

### 2. 可视化调试

```python
# 可视化模型结构
from torchviz import make_dot
dot = make_dot(output, params=dict(model.named_parameters()))
dot.render('model_graph')

# 可视化注意力权重
from src.visualization.debug_tools import plot_attention_debug
plot_attention_debug(attention_weights, 'debug_attention.png')
```

### 3. 性能分析

```python
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    output = model(input_tensor)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## 部署

### 1. 模型导出

```python
# 导出为ONNX格式
torch.onnx.export(model, dummy_input, "model.onnx")

# 导出为TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")
```

### 2. Docker 部署

```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "run_webapp.py"]
```

### 3. 生产环境配置

创建生产环境配置文件 `configs/production.yaml`：

```yaml
system:
    device: "cuda"
    mixed_precision: true

logging:
    level: "WARNING"

paths:
    models: "/data/models"
    logs: "/var/log/app"
```

## 性能优化

### 1. 模型优化

-   使用混合精度训练
-   模型量化
-   知识蒸馏
-   剪枝技术

### 2. 数据加载优化

```python
# 使用更多的worker
DataLoader(dataset, num_workers=8, pin_memory=True)

# 预取数据
DataLoader(dataset, prefetch_factor=2)
```

### 3. 内存优化

```python
# 使用梯度累积
for batch in data_loader:
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 故障排除

### 常见问题

1. **CUDA 内存不足**

    - 减小 batch_size
    - 使用梯度累积
    - 启用混合精度

2. **模型加载失败**

    - 检查模型路径
    - 验证模型结构匹配
    - 检查设备兼容性

3. **训练不收敛**
    - 调整学习率
    - 检查数据预处理
    - 验证损失函数

### 调试步骤

1. 启用详细日志
2. 检查数据加载
3. 验证模型输出形状
4. 测试小批量数据
5. 使用简化模型测试

## 文档贡献

### 1. 更新文档

文档使用 Markdown 格式，位于 `docs/` 目录：

```bash
# 预览文档（如果使用MkDocs）
mkdocs serve

# 构建文档
mkdocs build
```

### 2. API 文档

使用 docstring 记录函数和类：

```python
def train_model(model, data_loader, epochs):
    """训练模型

    Args:
        model (nn.Module): 待训练的模型
        data_loader (DataLoader): 数据加载器
        epochs (int): 训练轮数

    Returns:
        dict: 训练结果，包含损失和准确率

    Example:
        >>> model = ResNetSE(num_classes=10)
        >>> results = train_model(model, train_loader, 50)
    """
```

## 社区贡献

### 1. 报告问题

使用 GitHub Issues 报告 bug 或提出功能请求。

### 2. 贡献代码

1. Fork 项目
2. 创建功能分支
3. 编写代码和测试
4. 提交 Pull Request

### 3. 代码审查

所有代码提交都需要通过代码审查：

-   代码风格检查
-   功能测试验证
-   文档完整性检查
-   性能影响评估

## 资源链接

-   [PyTorch 官方文档](https://pytorch.org/docs/)
-   [Flask 文档](https://flask.palletsprojects.com/)
-   [注意力机制论文](https://arxiv.org/abs/1709.01507)
-   [CBAM 论文](https://arxiv.org/abs/1807.06521)
