# 训练指南

本文档介绍如何使用基线模型训练系统进行番茄叶斑病细粒度识别。

## 🚀 快速开始

### 1. 环境准备

确保已完成数据预处理：

```bash
python scripts/preprocess_data.py
python scripts/validate_preprocessing.py
```

### 2. 测试训练系统

运行快速训练测试（使用少量数据和 epoch）：

```bash
python scripts/quick_train.py
```

### 3. 开始完整训练

使用默认参数训练基线模型：

```bash
python scripts/train_baseline.py
```

## 📋 训练参数

### 基本参数

| 参数              | 默认值 | 说明                                    |
| ----------------- | ------ | --------------------------------------- |
| `--epochs`        | 100    | 训练轮数                                |
| `--learning_rate` | 0.001  | 学习率                                  |
| `--weight_decay`  | 1e-4   | 权重衰减                                |
| `--dropout_rate`  | 0.5    | Dropout 比率                            |
| `--optimizer`     | adam   | 优化器 (adam/sgd)                       |
| `--scheduler`     | step   | 学习率调度器 (step/cosine/plateau/none) |

### 早停参数

| 参数               | 默认值 | 说明         |
| ------------------ | ------ | ------------ |
| `--early_stopping` | False  | 是否启用早停 |
| `--patience`       | 15     | 早停容忍轮数 |
| `--min_delta`      | 0.001  | 最小改善幅度 |

### 其他参数

| 参数                | 默认值            | 说明             |
| ------------------- | ----------------- | ---------------- |
| `--experiment_name` | resnet50_baseline | 实验名称         |
| `--seed`            | 42                | 随机种子         |
| `--freeze_backbone` | False             | 是否冻结骨干网络 |
| `--resume`          | None              | 从检查点恢复训练 |

## 🎯 训练示例

### 基础训练

```bash
python scripts/train_baseline.py \
    --epochs 50 \
    --learning_rate 0.001 \
    --experiment_name "baseline_50epochs"
```

### 启用早停的训练

```bash
python scripts/train_baseline.py \
    --epochs 100 \
    --early_stopping \
    --patience 10 \
    --experiment_name "baseline_early_stop"
```

### 冻结骨干网络的训练

```bash
python scripts/train_baseline.py \
    --freeze_backbone \
    --epochs 30 \
    --learning_rate 0.01 \
    --experiment_name "baseline_frozen"
```

### 使用不同优化器和调度器

```bash
python scripts/train_baseline.py \
    --optimizer sgd \
    --scheduler cosine \
    --learning_rate 0.01 \
    --experiment_name "baseline_sgd_cosine"
```

### 从检查点恢复训练

```bash
python scripts/train_baseline.py \
    --resume "outputs/models/baseline_experiment/checkpoint_epoch_20.pth" \
    --experiment_name "baseline_resumed"
```

## 📊 输出文件

训练完成后，会在以下位置生成文件：

### 模型文件

- `outputs/models/{experiment_name}/`
  - `checkpoint_epoch_*.pth` - 定期保存的检查点
  - `best_checkpoint_epoch_*.pth` - 最佳模型检查点
  - `training_history.json` - 训练历史记录

### 日志文件

- `outputs/logs/{experiment_name}/`
  - TensorBoard 日志文件（如果安装了 tensorboard）

### 结果文件

- `outputs/results/{experiment_name}_results.json` - 训练结果摘要

## 📈 监控训练

### 1. 控制台输出

训练过程中会实时显示：

- 每个 epoch 的训练和验证损失/准确率
- 学习率变化
- 训练时间
- 最佳验证准确率

### 2. TensorBoard（可选）

如果安装了 tensorboard，可以可视化训练过程：

```bash
# 安装tensorboard
pip install tensorboard

# 启动TensorBoard
tensorboard --logdir outputs/logs
```

### 3. 日志文件

详细的训练日志保存在 `outputs/logs/` 目录中。

## 🔧 故障排除

### 常见问题

1. **内存不足**

   - 减小 batch_size：修改 `src/config/config.py` 中的 `BATCH_SIZE`
   - 减少 num_workers：修改 `NUM_WORKERS`

2. **训练速度慢**

   - 在 CPU 上训练较慢，考虑使用 GPU
   - 减少数据增强操作

3. **TensorBoard 警告**

   - 安装 tensorboard：`pip install tensorboard`
   - 或忽略警告，不影响训练

4. **模型不收敛**
   - 调整学习率
   - 检查数据质量
   - 尝试不同的优化器

### 调试模式

使用小数据集快速测试：

```bash
python scripts/quick_train.py
```

## 📝 实验记录

建议为每次实验记录：

- 实验目的
- 参数设置
- 最终结果
- 观察和结论

示例实验记录：

```
实验名称: baseline_50epochs
目的: 建立基线性能
参数: epochs=50, lr=0.001, optimizer=adam
结果: 最佳验证准确率 85.2%
观察: 在第35轮达到最佳性能，之后开始过拟合
结论: 建议使用早停机制
```

## 🎯 下一步

完成基线模型训练后，可以：

1. 分析训练结果和模型性能
2. 实现注意力机制模型
3. 进行模型对比和消融实验
4. 优化超参数
5. 进行模型集成
