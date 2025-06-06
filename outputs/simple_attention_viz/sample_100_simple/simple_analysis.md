# 简化注意力分析报告

## 样本信息
- 图像路径: sample_100
- 真实标签: bacterial_spot

## 模型预测对比
| 模型 | 预测类别 | 置信度 | 预测正确 |
|------|----------|--------|----------|
| SE-Net | bacterial_spot | 0.9283 | ✅ |
| CBAM | bacterial_spot | 0.9360 | ✅ |

## 详细概率分布

### SE-Net
- **bacterial_spot**: 0.9283 (92.83%)
- septoria_leaf_spot: 0.0283 (2.83%)
- target_spot: 0.0196 (1.96%)
- healthy: 0.0238 (2.38%)

### CBAM
- **bacterial_spot**: 0.9360 (93.60%)
- septoria_leaf_spot: 0.0216 (2.16%)
- target_spot: 0.0237 (2.37%)
- healthy: 0.0187 (1.87%)

## 注意力权重统计

### SE-Net
- 检测到 96 个注意力模块
  - layer1.0.se.global_avg_pool: 形状 [1, 256, 1, 1]
  - layer1.0.se.fc1: 形状 [1, 16]
  - layer1.0.se.relu: 形状 [1, 16]
  - layer1.0.se.fc2: 形状 [1, 256]
  - layer1.0.se.sigmoid: 形状 [1, 256]
  - layer1.0.se: 形状 [1, 256, 56, 56]
  - layer1.1.se.global_avg_pool: 形状 [1, 256, 1, 1]
  - layer1.1.se.fc1: 形状 [1, 16]
  - layer1.1.se.relu: 形状 [1, 16]
  - layer1.1.se.fc2: 形状 [1, 256]
  - layer1.1.se.sigmoid: 形状 [1, 256]
  - layer1.1.se: 形状 [1, 256, 56, 56]
  - layer1.2.se.global_avg_pool: 形状 [1, 256, 1, 1]
  - layer1.2.se.fc1: 形状 [1, 16]
  - layer1.2.se.relu: 形状 [1, 16]
  - layer1.2.se.fc2: 形状 [1, 256]
  - layer1.2.se.sigmoid: 形状 [1, 256]
  - layer1.2.se: 形状 [1, 256, 56, 56]
  - layer2.0.se.global_avg_pool: 形状 [1, 512, 1, 1]
  - layer2.0.se.fc1: 形状 [1, 32]
  - layer2.0.se.relu: 形状 [1, 32]
  - layer2.0.se.fc2: 形状 [1, 512]
  - layer2.0.se.sigmoid: 形状 [1, 512]
  - layer2.0.se: 形状 [1, 512, 28, 28]
  - layer2.1.se.global_avg_pool: 形状 [1, 512, 1, 1]
  - layer2.1.se.fc1: 形状 [1, 32]
  - layer2.1.se.relu: 形状 [1, 32]
  - layer2.1.se.fc2: 形状 [1, 512]
  - layer2.1.se.sigmoid: 形状 [1, 512]
  - layer2.1.se: 形状 [1, 512, 28, 28]
  - layer2.2.se.global_avg_pool: 形状 [1, 512, 1, 1]
  - layer2.2.se.fc1: 形状 [1, 32]
  - layer2.2.se.relu: 形状 [1, 32]
  - layer2.2.se.fc2: 形状 [1, 512]
  - layer2.2.se.sigmoid: 形状 [1, 512]
  - layer2.2.se: 形状 [1, 512, 28, 28]
  - layer2.3.se.global_avg_pool: 形状 [1, 512, 1, 1]
  - layer2.3.se.fc1: 形状 [1, 32]
  - layer2.3.se.relu: 形状 [1, 32]
  - layer2.3.se.fc2: 形状 [1, 512]
  - layer2.3.se.sigmoid: 形状 [1, 512]
  - layer2.3.se: 形状 [1, 512, 28, 28]
  - layer3.0.se.global_avg_pool: 形状 [1, 1024, 1, 1]
  - layer3.0.se.fc1: 形状 [1, 64]
  - layer3.0.se.relu: 形状 [1, 64]
  - layer3.0.se.fc2: 形状 [1, 1024]
  - layer3.0.se.sigmoid: 形状 [1, 1024]
  - layer3.0.se: 形状 [1, 1024, 14, 14]
  - layer3.1.se.global_avg_pool: 形状 [1, 1024, 1, 1]
  - layer3.1.se.fc1: 形状 [1, 64]
  - layer3.1.se.relu: 形状 [1, 64]
  - layer3.1.se.fc2: 形状 [1, 1024]
  - layer3.1.se.sigmoid: 形状 [1, 1024]
  - layer3.1.se: 形状 [1, 1024, 14, 14]
  - layer3.2.se.global_avg_pool: 形状 [1, 1024, 1, 1]
  - layer3.2.se.fc1: 形状 [1, 64]
  - layer3.2.se.relu: 形状 [1, 64]
  - layer3.2.se.fc2: 形状 [1, 1024]
  - layer3.2.se.sigmoid: 形状 [1, 1024]
  - layer3.2.se: 形状 [1, 1024, 14, 14]
  - layer3.3.se.global_avg_pool: 形状 [1, 1024, 1, 1]
  - layer3.3.se.fc1: 形状 [1, 64]
  - layer3.3.se.relu: 形状 [1, 64]
  - layer3.3.se.fc2: 形状 [1, 1024]
  - layer3.3.se.sigmoid: 形状 [1, 1024]
  - layer3.3.se: 形状 [1, 1024, 14, 14]
  - layer3.4.se.global_avg_pool: 形状 [1, 1024, 1, 1]
  - layer3.4.se.fc1: 形状 [1, 64]
  - layer3.4.se.relu: 形状 [1, 64]
  - layer3.4.se.fc2: 形状 [1, 1024]
  - layer3.4.se.sigmoid: 形状 [1, 1024]
  - layer3.4.se: 形状 [1, 1024, 14, 14]
  - layer3.5.se.global_avg_pool: 形状 [1, 1024, 1, 1]
  - layer3.5.se.fc1: 形状 [1, 64]
  - layer3.5.se.relu: 形状 [1, 64]
  - layer3.5.se.fc2: 形状 [1, 1024]
  - layer3.5.se.sigmoid: 形状 [1, 1024]
  - layer3.5.se: 形状 [1, 1024, 14, 14]
  - layer4.0.se.global_avg_pool: 形状 [1, 2048, 1, 1]
  - layer4.0.se.fc1: 形状 [1, 128]
  - layer4.0.se.relu: 形状 [1, 128]
  - layer4.0.se.fc2: 形状 [1, 2048]
  - layer4.0.se.sigmoid: 形状 [1, 2048]
  - layer4.0.se: 形状 [1, 2048, 7, 7]
  - layer4.1.se.global_avg_pool: 形状 [1, 2048, 1, 1]
  - layer4.1.se.fc1: 形状 [1, 128]
  - layer4.1.se.relu: 形状 [1, 128]
  - layer4.1.se.fc2: 形状 [1, 2048]
  - layer4.1.se.sigmoid: 形状 [1, 2048]
  - layer4.1.se: 形状 [1, 2048, 7, 7]
  - layer4.2.se.global_avg_pool: 形状 [1, 2048, 1, 1]
  - layer4.2.se.fc1: 形状 [1, 128]
  - layer4.2.se.relu: 形状 [1, 128]
  - layer4.2.se.fc2: 形状 [1, 2048]
  - layer4.2.se.sigmoid: 形状 [1, 2048]
  - layer4.2.se: 形状 [1, 2048, 7, 7]

### CBAM
- 检测到 16 个注意力模块
  - layer1.0.cbam: 形状 [1, 256, 56, 56]
  - layer1.1.cbam: 形状 [1, 256, 56, 56]
  - layer1.2.cbam: 形状 [1, 256, 56, 56]
  - layer2.0.cbam: 形状 [1, 512, 28, 28]
  - layer2.1.cbam: 形状 [1, 512, 28, 28]
  - layer2.2.cbam: 形状 [1, 512, 28, 28]
  - layer2.3.cbam: 形状 [1, 512, 28, 28]
  - layer3.0.cbam: 形状 [1, 1024, 14, 14]
  - layer3.1.cbam: 形状 [1, 1024, 14, 14]
  - layer3.2.cbam: 形状 [1, 1024, 14, 14]
  - layer3.3.cbam: 形状 [1, 1024, 14, 14]
  - layer3.4.cbam: 形状 [1, 1024, 14, 14]
  - layer3.5.cbam: 形状 [1, 1024, 14, 14]
  - layer4.0.cbam: 形状 [1, 2048, 7, 7]
  - layer4.1.cbam: 形状 [1, 2048, 7, 7]
  - layer4.2.cbam: 形状 [1, 2048, 7, 7]
