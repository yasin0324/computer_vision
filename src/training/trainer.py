"""
训练器类：负责模型训练和验证
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, Any, Optional, Tuple, List
from tqdm import tqdm

# 可选导入TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

from .utils import EarlyStopping, AverageMeter, save_checkpoint, accuracy, Timer, format_time
from ..utils.logger import get_logger


class Trainer:
    """
    训练器类
    
    负责模型的训练、验证、保存和日志记录
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cpu',
        save_dir: str = 'outputs/models',
        log_dir: str = 'outputs/logs',
        experiment_name: str = 'baseline_experiment'
    ):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            criterion: 损失函数
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 设备
            save_dir: 模型保存目录
            log_dir: 日志保存目录
            experiment_name: 实验名称
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        # 创建保存目录
        self.save_dir = Path(save_dir) / experiment_name
        self.log_dir = Path(log_dir) / experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = get_logger(f"trainer_{experiment_name}")
        
        # 设置TensorBoard（如果可用）
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        else:
            self.writer = None
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        self.logger.info(f"Trainer initialized for experiment: {experiment_name}")
        self.logger.info(f"Device: {device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        if not TENSORBOARD_AVAILABLE:
            self.logger.warning("TensorBoard not available - logging will be limited to console and files")
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        训练一个epoch
        
        Returns:
            (平均损失, 平均准确率)
        """
        self.model.train()
        
        losses = AverageMeter()
        accuracies = AverageMeter()
        
        # 进度条
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1} [Train]')
        
        for batch_idx, (images, targets, _) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            batch_size = images.size(0)
            
            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 计算准确率
            acc = accuracy(outputs, targets)[0]
            
            # 更新统计
            losses.update(loss.item(), batch_size)
            accuracies.update(acc.item(), batch_size)
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc': f'{accuracies.avg:.2f}%'
            })
            
            # 记录到TensorBoard（如果可用）
            if self.writer is not None:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
                self.writer.add_scalar('Train/BatchAcc', acc.item(), global_step)
        
        return losses.avg, accuracies.avg
    
    def validate_epoch(self) -> Tuple[float, float]:
        """
        验证一个epoch
        
        Returns:
            (平均损失, 平均准确率)
        """
        self.model.eval()
        
        losses = AverageMeter()
        accuracies = AverageMeter()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch + 1} [Val]')
            
            for images, targets, _ in pbar:
                images = images.to(self.device)
                targets = targets.to(self.device)
                batch_size = images.size(0)
                
                # 前向传播
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                # 计算准确率
                acc = accuracy(outputs, targets)[0]
                
                # 更新统计
                losses.update(loss.item(), batch_size)
                accuracies.update(acc.item(), batch_size)
                
                # 更新进度条
                pbar.set_postfix({
                    'Loss': f'{losses.avg:.4f}',
                    'Acc': f'{accuracies.avg:.2f}%'
                })
        
        return losses.avg, accuracies.avg
    
    def train(
        self,
        epochs: int,
        early_stopping: Optional[EarlyStopping] = None,
        save_every: int = 10
    ) -> Dict[str, List[float]]:
        """
        训练模型
        
        Args:
            epochs: 训练轮数
            early_stopping: 早停机制
            save_every: 每隔多少epoch保存一次模型
            
        Returns:
            训练历史
        """
        self.logger.info(f"Starting training for {epochs} epochs")
        
        total_timer = Timer()
        total_timer.start()
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_timer = Timer()
            epoch_timer.start()
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc = self.validate_epoch()
            
            # 学习率调度
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 记录历史
            current_lr = self.optimizer.param_groups[0]['lr']
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['learning_rates'].append(current_lr)
            
            # 记录到TensorBoard（如果可用）
            if self.writer is not None:
                self.writer.add_scalar('Epoch/TrainLoss', train_loss, epoch)
                self.writer.add_scalar('Epoch/TrainAcc', train_acc, epoch)
                self.writer.add_scalar('Epoch/ValLoss', val_loss, epoch)
                self.writer.add_scalar('Epoch/ValAcc', val_acc, epoch)
                self.writer.add_scalar('Epoch/LearningRate', current_lr, epoch)
            
            # 计算epoch时间
            epoch_time = epoch_timer.stop()
            
            # 日志记录
            self.logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
                f"LR: {current_lr:.6f} - Time: {format_time(epoch_time)}"
            )
            
            # 保存最佳模型
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.logger.info(f"New best validation accuracy: {val_acc:.2f}%")
            
            # 定期保存模型
            if (epoch + 1) % save_every == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # 早停检查
            if early_stopping is not None:
                if early_stopping(val_acc, self.model):
                    self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
        
        total_time = total_timer.stop()
        self.logger.info(f"Training completed in {format_time(total_time)}")
        self.logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
        # 保存训练历史
        self.save_training_history()
        
        # 关闭TensorBoard writer
        if self.writer is not None:
            self.writer.close()
        
        return self.train_history
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'train_history': self.train_history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存当前检查点
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch + 1}.pth'
        save_checkpoint(checkpoint, str(checkpoint_path), is_best)
        
        if is_best:
            self.logger.info(f"Best model saved to {self.save_dir / f'best_checkpoint_epoch_{epoch + 1}.pth'}")
    
    def save_training_history(self):
        """保存训练历史"""
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        self.logger.info(f"Training history saved to {history_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点文件路径
            
        Returns:
            加载的epoch数
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.train_history = checkpoint.get('train_history', {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'learning_rates': []
        })
        
        epoch = checkpoint['epoch']
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {epoch}")
        
        return epoch 