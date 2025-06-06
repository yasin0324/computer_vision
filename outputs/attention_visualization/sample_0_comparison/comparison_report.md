# 注意力机制对比分析报告

## 样本信息
- 真实标签: bacterial_spot
- 图像尺寸: torch.Size([1, 3, 224, 224])

## 模型预测对比
| 模型 | 预测类别 | 置信度 | 预测正确 |
|------|----------|--------|----------|
| SE-Net | bacterial_spot | 0.9222 | ✅ |
| CBAM | bacterial_spot | 0.9274 | ✅ |

## 详细概率分布

### SE-Net
- **bacterial_spot**: 0.9222 (92.22%)
- septoria_leaf_spot: 0.0239 (2.39%)
- target_spot: 0.0292 (2.92%)
- healthy: 0.0247 (2.47%)

### CBAM
- **bacterial_spot**: 0.9274 (92.74%)
- septoria_leaf_spot: 0.0253 (2.53%)
- target_spot: 0.0265 (2.65%)
- healthy: 0.0209 (2.09%)

## 注意力机制分析
### SE-Net (通道注意力)
- 关注特征通道的重要性
- 学习'关注什么'特征
- 参数较少，计算高效

### CBAM (双重注意力)
- 结合通道注意力和空间注意力
- 学习'关注什么'和'关注哪里'
- 更全面的注意力机制

## 可视化文件说明
- `original_image.png`: 原始输入图像
- `channel_attention.png`: 通道注意力权重分布
- `spatial_attention.png`: 空间注意力热力图（仅CBAM）
- `feature_maps.png`: 特征图可视化
- `grad_cam_*.png`: Grad-CAM类激活图
- `attention_analysis.md`: 详细注意力分析
