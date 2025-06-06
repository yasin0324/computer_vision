# 注意力机制可视化分析总结

## 实验概述
本实验对比分析了SE-Net和CBAM两种注意力机制在植物叶片病害识别任务中的表现。

## 模型信息
- **SE-Net**: ResNet50 + SE-Net注意力机制
- **CBAM**: ResNet50 + CBAM注意力机制
- **数据集**: 番茄叶斑病细粒度识别（4类）
- **类别**: bacterial_spot, septoria_leaf_spot, target_spot, healthy

## 注意力机制对比
### SE-Net (Squeeze-and-Excitation)
- **机制**: 通道注意力
- **原理**: 通过全局平均池化和全连接层学习通道权重
- **优势**: 参数少，计算高效，关注重要特征通道
- **可视化**: 通道权重分布图

### CBAM (Convolutional Block Attention Module)
- **机制**: 通道注意力 + 空间注意力
- **原理**: 先学习通道权重，再学习空间位置权重
- **优势**: 更全面的注意力，同时关注'什么'和'哪里'
- **可视化**: 通道权重分布图 + 空间注意力热力图

## 可视化内容说明
### 文件结构
```
attention_visualization/
├── sample_X_comparison/          # 单样本对比分析
│   ├── SE-Net/                   # SE-Net可视化结果
│   ├── CBAM/                     # CBAM可视化结果
│   └── comparison_report.md      # 对比分析报告
├── class_XXX/                    # 各类别批量分析
│   └── sample_X/
│       ├── SE-Net/
│       └── CBAM/
└── visualization_summary.md      # 总结报告
```

### 可视化图像说明
- **original_image.png**: 原始输入图像
- **channel_attention.png**: 通道注意力权重柱状图
- **spatial_attention.png**: 空间注意力热力图（仅CBAM）
- **feature_maps.png**: 不同层的特征图可视化
- **grad_cam_*.png**: Grad-CAM类激活图
- **attention_analysis.md**: 详细数值分析

## 分析要点
1. **通道注意力对比**: 观察SE-Net和CBAM的通道权重分布差异
2. **空间注意力分析**: CBAM的空间注意力如何定位关键区域
3. **预测一致性**: 两种模型在相同样本上的预测差异
4. **类别特异性**: 不同病害类别激活的注意力模式
5. **错误案例分析**: 模型预测错误时的注意力模式

## 使用建议
1. 首先查看单样本对比分析，理解两种注意力机制的差异
2. 浏览各类别的批量分析，发现类别特异的注意力模式
3. 重点关注预测错误的样本，分析注意力机制的局限性
4. 结合Grad-CAM和注意力权重，全面理解模型决策过程
