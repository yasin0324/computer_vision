#!/usr/bin/env python3
"""
测试基线模型是否能正常工作
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
from src.models.baseline import create_resnet_baseline
from src.config import config


def test_model():
    """测试模型创建和前向传播"""
    print("Testing baseline model...")
    
    # 创建模型
    model = create_resnet_baseline(
        num_classes=config.NUM_CLASSES,
        pretrained=config.PRETRAINED,
        dropout_rate=0.5,
        freeze_backbone=False
    )
    
    # 打印模型信息
    model_info = model.get_model_info()
    print("\nModel Information:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # 测试前向传播
    print("\nTesting forward pass...")
    model.eval()
    
    # 创建随机输入
    batch_size = 4
    x = torch.randn(batch_size, 3, config.INPUT_SIZE, config.INPUT_SIZE)
    print(f"Input shape: {x.shape}")
    
    # 前向传播
    with torch.no_grad():
        logits = model(x)
        print(f"Output shape: {logits.shape}")
        print(f"Output range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
        
        # 计算概率
        probs = torch.softmax(logits, dim=1)
        print(f"Probabilities shape: {probs.shape}")
        print(f"Probability sums: {probs.sum(dim=1)}")
        
        # 预测类别
        predictions = torch.argmax(logits, dim=1)
        print(f"Predictions: {predictions}")
    
    # 测试特征提取
    print("\nTesting feature extraction...")
    features = model.get_features(x)
    print(f"Features shape: {features.shape}")
    
    print("\n✅ Model test completed successfully!")


if __name__ == "__main__":
    test_model() 