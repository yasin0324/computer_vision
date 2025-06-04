#!/usr/bin/env python3
"""
基线模型训练脚本

使用ResNet50训练番茄叶斑病细粒度识别模型
"""

import sys
import os
from pathlib import Path
import json
import argparse

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import config
from src.models.baseline import create_resnet_baseline
from src.data.preprocessing import TomatoSpotDataset, get_data_transforms
from src.training.trainer import Trainer
from src.training.utils import EarlyStopping, set_seed
from src.utils.logger import setup_logger
import pandas as pd


def load_data() -> tuple:
    """
    加载预处理后的数据
    
    Returns:
        (train_loader, val_loader, test_loader, class_mapping)
    """
    print("Loading preprocessed data...")
    
    # 加载数据划分
    train_df = pd.read_csv(f"{config.PROCESSED_DATA_DIR}/train_split.csv")
    val_df = pd.read_csv(f"{config.PROCESSED_DATA_DIR}/val_split.csv")
    test_df = pd.read_csv(f"{config.PROCESSED_DATA_DIR}/test_split.csv")
    
    # 加载类别映射
    with open(f"{config.PROCESSED_DATA_DIR}/class_mapping.json", 'r') as f:
        class_mapping = json.load(f)
    
    print(f"Data loaded:")
    print(f"  Training samples: {len(train_df)}")
    print(f"  Validation samples: {len(val_df)}")
    print(f"  Test samples: {len(test_df)}")
    print(f"  Classes: {class_mapping['class_names']}")
    
    # 创建数据变换
    train_transform = get_data_transforms(config.INPUT_SIZE, augment=True)
    val_test_transform = get_data_transforms(config.INPUT_SIZE, augment=False)
    
    # 创建数据集
    train_dataset = TomatoSpotDataset(train_df, train_transform, class_mapping['label_to_idx'])
    val_dataset = TomatoSpotDataset(val_df, val_test_transform, class_mapping['label_to_idx'])
    test_dataset = TomatoSpotDataset(test_df, val_test_transform, class_mapping['label_to_idx'])
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_mapping


def create_model_and_optimizer(args) -> tuple:
    """
    创建模型和优化器
    
    Args:
        args: 命令行参数
        
    Returns:
        (model, criterion, optimizer, scheduler)
    """
    print("Creating model and optimizer...")
    
    # 创建模型
    model = create_resnet_baseline(
        num_classes=config.NUM_CLASSES,
        pretrained=config.PRETRAINED,
        dropout_rate=args.dropout_rate,
        freeze_backbone=args.freeze_backbone
    )
    
    # 打印模型信息
    model_info = model.get_model_info()
    print("Model Information:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 优化器
    if args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    # 学习率调度器
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma
        )
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs
        )
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
    else:
        scheduler = None
    
    return model, criterion, optimizer, scheduler


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Train baseline ResNet50 model')
    
    # 模型参数
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone weights')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=config.EPOCHS, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=config.LEARNING_RATE, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=config.WEIGHT_DECAY, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='step', choices=['step', 'cosine', 'plateau', 'none'], help='Learning rate scheduler')
    parser.add_argument('--step_size', type=int, default=config.STEP_SIZE, help='Step size for StepLR')
    parser.add_argument('--gamma', type=float, default=config.GAMMA, help='Gamma for StepLR')
    
    # 早停参数
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=config.EARLY_STOPPING['patience'], help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=config.EARLY_STOPPING['min_delta'], help='Early stopping min delta')
    
    # 其他参数
    parser.add_argument('--experiment_name', type=str, default='resnet50_baseline', help='Experiment name')
    parser.add_argument('--seed', type=int, default=config.RANDOM_STATE, help='Random seed')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置日志
    logger = setup_logger(f"train_{args.experiment_name}")
    logger.info("Starting baseline model training")
    logger.info(f"Arguments: {vars(args)}")
    
    # 检查设备
    device = config.DEVICE
    logger.info(f"Using device: {device}")
    
    try:
        # 加载数据
        train_loader, val_loader, test_loader, class_mapping = load_data()
        
        # 创建模型和优化器
        model, criterion, optimizer, scheduler = create_model_and_optimizer(args)
        
        # 创建训练器
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
            experiment_name=args.experiment_name
        )
        
        # 设置早停
        early_stopping = None
        if args.early_stopping:
            early_stopping = EarlyStopping(
                patience=args.patience,
                min_delta=args.min_delta,
                restore_best_weights=True
            )
            logger.info(f"Early stopping enabled with patience={args.patience}")
        
        # 从检查点恢复（如果指定）
        start_epoch = 0
        if args.resume:
            start_epoch = trainer.load_checkpoint(args.resume)
            logger.info(f"Resumed from checkpoint: {args.resume}")
        
        # 开始训练
        logger.info("Starting training...")
        train_history = trainer.train(
            epochs=args.epochs - start_epoch,
            early_stopping=early_stopping,
            save_every=10
        )
        
        # 保存最终结果
        results = {
            'experiment_name': args.experiment_name,
            'model_info': model.get_model_info(),
            'training_args': vars(args),
            'best_val_acc': trainer.best_val_acc,
            'final_train_acc': train_history['train_acc'][-1] if train_history['train_acc'] else 0,
            'final_val_acc': train_history['val_acc'][-1] if train_history['val_acc'] else 0,
            'total_epochs': len(train_history['train_acc'])
        }
        
        results_path = f"outputs/results/{args.experiment_name}_results.json"
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
        logger.info(f"Results saved to: {results_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 