#!/usr/bin/env python3
"""
改进的基线模型训练脚本 - 解决过拟合问题
主要改进：
1. 更强的正则化（dropout=0.7, weight_decay=0.001）
2. 更强的数据增强
3. 更严格的早停机制
4. 减少训练轮数
5. 使用AdamW优化器和标签平滑
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

from src.models.baseline import create_resnet_baseline
from src.data.dataset import TomatoSpotDataset
from src.training.trainer import Trainer
from src.config.config import config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Train improved baseline model')
    
    # 改进的默认参数
    parser.add_argument('--dropout_rate', type=float, default=0.7, help='Higher dropout rate')
    parser.add_argument('--epochs', type=int, default=50, help='Reduced epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Lower learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Higher weight decay')
    parser.add_argument('--patience', type=int, default=8, help='Stricter early stopping')
    parser.add_argument('--experiment_name', type=str, default='resnet50_baseline_improved')
    
    args = parser.parse_args()
    
    print("Starting improved baseline training with anti-overfitting measures...")
    print(f"Key improvements:")
    print(f"  - Dropout rate: {args.dropout_rate} (was 0.5)")
    print(f"  - Epochs: {args.epochs} (was 100)")
    print(f"  - Learning rate: {args.learning_rate} (was 0.001)")
    print(f"  - Weight decay: {args.weight_decay} (was 0.0001)")
    print(f"  - Early stopping patience: {args.patience} (was 15)")
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 改进的数据增强
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
    
    print(f"Data loaded: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    # 创建改进的模型
    model = create_resnet_baseline(
        num_classes=config.NUM_CLASSES,
        pretrained=True,
        dropout_rate=args.dropout_rate,
        freeze_backbone=False
    )
    model.to(device)
    
    # 改进的训练组件
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
    print("Starting training...")
    trainer.train(epochs=args.epochs, early_stopping=early_stopping)
    
    print("Training completed!")
    print(f"Best validation accuracy: {trainer.best_val_acc:.4f}%")


if __name__ == "__main__":
    main() 