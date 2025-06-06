#!/usr/bin/env python3
"""
CBAM注意力机制训练脚本
基于SE-Net的成功经验，实现CBAM（Convolutional Block Attention Module）
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from torchvision import transforms

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.attention_models import create_resnet_with_attention, get_model_info
from src.data.dataset import TomatoSpotDataset
from src.training.trainer import Trainer
from src.config.config import config


def load_baseline_weights_to_cbam_model(cbam_model, baseline_checkpoint_path):
    """
    将基线模型的权重加载到CBAM模型中
    
    Args:
        cbam_model: CBAM模型
        baseline_checkpoint_path: 基线模型检查点路径
    """
    # 加载基线检查点
    checkpoint = torch.load(baseline_checkpoint_path, map_location='cpu')
    baseline_state_dict = checkpoint['model_state_dict']
    
    # 获取CBAM模型的状态字典
    cbam_state_dict = cbam_model.state_dict()
    
    # 映射基线模型的权重到CBAM模型
    weight_mapping = {}
    
    # 映射backbone的权重
    for key in baseline_state_dict:
        if key.startswith('backbone.'):
            # 移除backbone前缀
            new_key = key.replace('backbone.', '')
            
            # 处理第一层卷积和BN
            if new_key.startswith('0.'):
                new_key = new_key.replace('0.', 'conv1.')
            elif new_key.startswith('1.'):
                new_key = new_key.replace('1.', 'bn1.')
            # 处理ResNet层
            elif new_key.startswith('4.'):
                new_key = new_key.replace('4.', 'layer1.')
            elif new_key.startswith('5.'):
                new_key = new_key.replace('5.', 'layer2.')
            elif new_key.startswith('6.'):
                new_key = new_key.replace('6.', 'layer3.')
            elif new_key.startswith('7.'):
                new_key = new_key.replace('7.', 'layer4.')
            
            weight_mapping[new_key] = baseline_state_dict[key]
    
    # 加载权重到CBAM模型（忽略CBAM相关的参数和分类器）
    missing_keys = []
    loaded_keys = []
    
    for key in weight_mapping:
        if key in cbam_state_dict:
            # 检查形状是否匹配
            if weight_mapping[key].shape == cbam_state_dict[key].shape:
                cbam_state_dict[key] = weight_mapping[key]
                loaded_keys.append(key)
            else:
                missing_keys.append(f"{key} (shape mismatch: {weight_mapping[key].shape} vs {cbam_state_dict[key].shape})")
        else:
            missing_keys.append(key)
    
    # 加载权重
    cbam_model.load_state_dict(cbam_state_dict, strict=False)
    
    print(f"成功加载基线模型权重")
    print(f"已加载的权重: {len(loaded_keys)} 个")
    if missing_keys:
        print(f"跳过的权重: {len(missing_keys)} 个")
    print("注意: CBAM模块和分类器权重将从头开始训练")
    
    return cbam_model


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Train CBAM attention model')
    
    # CBAM特定参数
    parser.add_argument('--reduction', type=int, default=16, help='CBAM reduction ratio')
    parser.add_argument('--dropout_rate', type=float, default=0.7, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--patience', type=int, default=8, help='Early stopping patience')
    parser.add_argument('--experiment_name', type=str, default='resnet50_cbam')
    parser.add_argument('--load_baseline', type=str, default=None, 
                       help='Path to baseline checkpoint to initialize from')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🔥 CBAM 注意力机制训练开始")
    print("="*60)
    print(f"实验配置:")
    print(f"  - 注意力类型: CBAM (Convolutional Block Attention Module)")
    print(f"  - 降维比例: {args.reduction}")
    print(f"  - Dropout率: {args.dropout_rate}")
    print(f"  - 训练轮数: {args.epochs}")
    print(f"  - 学习率: {args.learning_rate}")
    print(f"  - 权重衰减: {args.weight_decay}")
    print(f"  - 早停patience: {args.patience}")
    print(f"  - 实验名称: {args.experiment_name}")
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  - 使用设备: {device}")
    
    # 数据增强（与SE-Net保持一致）
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(config.INPUT_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据
    data_dir = Path("data/processed")
    train_df = pd.read_csv(data_dir / "train_split.csv")
    val_df = pd.read_csv(data_dir / "val_split.csv")
    
    class_names = list(config.TARGET_CLASSES.values())
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    
    train_dataset = TomatoSpotDataset(train_df, train_transform, label_to_idx)
    val_dataset = TomatoSpotDataset(val_df, val_transform, label_to_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, 
                             num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                           num_workers=config.NUM_WORKERS, pin_memory=True)
    
    print(f"数据加载完成: {len(train_dataset)} 训练样本, {len(val_dataset)} 验证样本")
    
    # 创建CBAM模型
    print("创建CBAM模型...")
    model = create_resnet_with_attention(
        num_classes=config.NUM_CLASSES,
        attention_type='cbam',
        reduction=args.reduction,
        dropout_rate=args.dropout_rate,
        pretrained=True  # 先使用预训练权重
    )
    
    # 如果指定了基线模型，则加载其权重
    if args.load_baseline:
        print(f"从基线模型初始化权重: {args.load_baseline}")
        model = load_baseline_weights_to_cbam_model(model, args.load_baseline)
    else:
        print("使用ImageNet预训练权重初始化")
    
    model.to(device)
    
    # 打印模型信息
    model_info = get_model_info(model)
    print(f"\n模型信息:")
    print(f"  - 总参数数: {model_info['total_parameters']:,}")
    print(f"  - 可训练参数: {model_info['trainable_parameters']:,}")
    print(f"  - CBAM模块数量: {model_info['cbam_blocks']}")
    print(f"  - 模型大小: {model_info['model_size_mb']:.2f} MB")
    
    # 训练组件（与SE-Net保持一致）
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, 
                           weight_decay=args.weight_decay)  # AdamW
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        experiment_name=args.experiment_name
    )
    
    # 创建早停机制
    from src.training.utils import EarlyStopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=0.005
    )
    
    # 开始训练
    print("\n🚀 开始CBAM训练...")
    print("="*60)
    trainer.train(epochs=args.epochs, early_stopping=early_stopping)
    
    print("\n🎉 CBAM训练完成!")
    print(f"最佳验证准确率: {trainer.best_val_acc:.4f}%")
    
    # 保存模型信息
    info_path = Path(f"outputs/models/{args.experiment_name}/model_info.json")
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"模型信息已保存到: {info_path}")


if __name__ == "__main__":
    main() 