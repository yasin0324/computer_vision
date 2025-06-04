#!/usr/bin/env python3
"""
快速训练脚本 - 用于测试训练流程

运行少量epoch来验证训练系统是否正常工作
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import json

from src.config import config
from src.models.baseline import create_resnet_baseline
from src.data.preprocessing import TomatoSpotDataset, get_data_transforms
from src.training.trainer import Trainer
from src.training.utils import EarlyStopping, set_seed
from src.utils.logger import setup_logger


def main():
    """快速训练测试"""
    print("=== Quick Training Test ===")
    
    # 设置随机种子
    set_seed(42)
    
    # 设置日志
    logger = setup_logger("quick_train_test")
    logger.info("Starting quick training test")
    
    # 检查设备
    device = config.DEVICE
    print(f"Using device: {device}")
    
    try:
        # 加载数据
        print("\n1. Loading data...")
        train_df = pd.read_csv(f"{config.PROCESSED_DATA_DIR}/train_split.csv")
        val_df = pd.read_csv(f"{config.PROCESSED_DATA_DIR}/val_split.csv")
        
        with open(f"{config.PROCESSED_DATA_DIR}/class_mapping.json", 'r') as f:
            class_mapping = json.load(f)
        
        # 使用小批量数据进行快速测试
        train_df_small = train_df.head(100)  # 只用100个训练样本
        val_df_small = val_df.head(50)       # 只用50个验证样本
        
        print(f"Training samples: {len(train_df_small)}")
        print(f"Validation samples: {len(val_df_small)}")
        
        # 创建数据变换
        train_transform = get_data_transforms(config.INPUT_SIZE, augment=True)
        val_transform = get_data_transforms(config.INPUT_SIZE, augment=False)
        
        # 创建数据集
        train_dataset = TomatoSpotDataset(train_df_small, train_transform, class_mapping['label_to_idx'])
        val_dataset = TomatoSpotDataset(val_df_small, val_transform, class_mapping['label_to_idx'])
        
        # 创建数据加载器（使用小批次）
        train_loader = DataLoader(
            train_dataset, 
            batch_size=8,  # 小批次
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        # 创建模型
        print("\n2. Creating model...")
        model = create_resnet_baseline(
            num_classes=config.NUM_CLASSES,
            pretrained=config.PRETRAINED,
            dropout_rate=0.5,
            freeze_backbone=False
        )
        
        model_info = model.get_model_info()
        print("Model Information:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        # 创建损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        # 创建训练器
        print("\n3. Creating trainer...")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            save_dir='outputs/models',
            log_dir='outputs/logs',
            experiment_name='quick_test'
        )
        
        # 设置早停（较小的patience）
        early_stopping = EarlyStopping(
            patience=3,
            min_delta=0.001,
            restore_best_weights=True
        )
        
        # 开始训练（只训练5个epoch）
        print("\n4. Starting training...")
        train_history = trainer.train(
            epochs=5,
            early_stopping=early_stopping,
            save_every=2
        )
        
        # 打印结果
        print("\n5. Training Results:")
        print(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
        if train_history['train_acc']:
            print(f"Final training accuracy: {train_history['train_acc'][-1]:.2f}%")
            print(f"Final validation accuracy: {train_history['val_acc'][-1]:.2f}%")
        
        print("\n✅ Quick training test completed successfully!")
        
    except Exception as e:
        logger.error(f"Quick training test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 