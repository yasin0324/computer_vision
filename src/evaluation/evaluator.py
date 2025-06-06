#!/usr/bin/env python3
"""
模型评估器模块

提供基础模型和注意力模型的评估功能
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.baseline import create_resnet_baseline
from src.models.attention_models import create_senet_model, create_cbam_model
from src.data.dataset import TomatoSpotDataset
from src.config.config import config


class BaseEvaluator:
    """基础评估器类"""
    
    def __init__(self, model_path: str, device: str = None):
        """
        初始化评估器
        
        Args:
            model_path: 训练好的模型权重路径
            device: 计算设备
        """
        self.model_path = model_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 定义测试时的图像变换
        self.test_transform = transforms.Compose([
            transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 类别映射
        self.class_names = list(config.TARGET_CLASSES.values())
        self.label_to_idx = {label: idx for idx, label in enumerate(self.class_names)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # 加载模型
        self.model = self._load_model()
        
    def _load_model(self) -> torch.nn.Module:
        """加载训练好的模型 - 子类需要实现"""
        raise NotImplementedError("Subclasses must implement _load_model method")
        
    def _load_test_data(self) -> DataLoader:
        """加载测试数据"""
        print("Loading test data...")
        
        # 读取测试集CSV文件
        test_csv_path = Path(project_root) / "data" / "processed" / "test_split.csv"
        test_df = pd.read_csv(test_csv_path)
        
        print(f"Test dataset size: {len(test_df)}")
        print(f"Test classes distribution:")
        for class_name, count in test_df['label'].value_counts().items():
            print(f"  {class_name}: {count}")
        
        # 创建数据集
        test_dataset = TomatoSpotDataset(
            dataframe=test_df,
            transform=self.test_transform,
            label_to_idx=self.label_to_idx
        )
        
        # 创建数据加载器
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        return test_loader
    
    def evaluate(self, save_predictions: bool = True) -> Dict[str, Any]:
        """在测试集上评估模型"""
        print("Starting model evaluation...")
        
        # 加载测试数据
        test_loader = self._load_test_data()
        
        # 存储预测结果
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_image_paths = []
        
        # 评估模型
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images, labels, image_paths) in enumerate(tqdm(test_loader, desc="Evaluating")):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                outputs = self._forward_pass(images)
                logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
                
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                # 收集结果
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_image_paths.extend(image_paths)
        
        # 转换为numpy数组
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # 计算评估指标
        results = self._calculate_metrics(all_labels, all_predictions, all_probabilities)
        
        if save_predictions:
            # 生成混淆矩阵
            self._plot_confusion_matrix(all_labels, all_predictions, results['output_dir'])
            
            # 保存详细结果
            self._save_detailed_results(
                all_labels, all_predictions, all_probabilities, 
                all_image_paths, results['output_dir']
            )
        
        return results
    
    def _forward_pass(self, images: torch.Tensor) -> Union[torch.Tensor, Tuple]:
        """前向传播 - 子类可以重写以处理不同的输出格式"""
        return self.model(images)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_prob: np.ndarray) -> Dict[str, Any]:
        """计算评估指标"""
        
        # 基本指标
        accuracy = accuracy_score(y_true, y_pred)
        
        # 每个类别的精确率、召回率、F1分数
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(len(self.class_names))
        )
        
        # 宏平均和微平均
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='micro'
        )
        
        # 分类报告
        class_report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # 创建输出目录
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        model_name = self._get_model_name()
        output_dir = Path(project_root) / "outputs" / "evaluation" / f"{model_name}_evaluation_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 组织结果
        results = {
            'model_name': model_name,
            'model_path': self.model_path,
            'test_accuracy': float(accuracy),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1),
            'micro_precision': float(micro_precision),
            'micro_recall': float(micro_recall),
            'micro_f1': float(micro_f1),
            'per_class_metrics': {
                self.class_names[i]: {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1_score': float(f1[i]),
                    'support': int(support[i])
                }
                for i in range(len(self.class_names))
            },
            'classification_report': class_report,
            'output_dir': str(output_dir),
            'total_samples': len(y_true),
            'correct_predictions': int(np.sum(y_true == y_pred)),
            'incorrect_predictions': int(np.sum(y_true != y_pred))
        }
        
        return results
    
    def _get_model_name(self) -> str:
        """获取模型名称 - 子类可以重写"""
        return "baseline"
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              output_dir: str):
        """绘制并保存混淆矩阵"""
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 计算百分比混淆矩阵
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 绘制数量混淆矩阵
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=ax1)
        ax1.set_title('Confusion Matrix (Counts)')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        
        # 绘制百分比混淆矩阵
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=ax2)
        ax2.set_title('Confusion Matrix (Percentages)')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        
        plt.tight_layout()
        
        # 保存图形
        confusion_matrix_path = Path(output_dir) / "confusion_matrix.png"
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to: {confusion_matrix_path}")
        
        # 保存混淆矩阵数据
        cm_data = {
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_percent': cm_percent.tolist(),
            'class_names': self.class_names
        }
        
        cm_json_path = Path(output_dir) / "confusion_matrix.json"
        with open(cm_json_path, 'w') as f:
            json.dump(cm_data, f, indent=2)
    
    def _save_detailed_results(self, y_true: np.ndarray, y_pred: np.ndarray,
                              y_prob: np.ndarray, image_paths: List[str],
                              output_dir: str):
        """保存详细的预测结果"""
        
        # 创建详细结果DataFrame
        results_df = pd.DataFrame({
            'image_path': image_paths,
            'true_label': [self.idx_to_label[idx] for idx in y_true],
            'predicted_label': [self.idx_to_label[idx] for idx in y_pred],
            'correct': y_true == y_pred,
            'confidence': np.max(y_prob, axis=1)
        })
        
        # 添加每个类别的概率
        for i, class_name in enumerate(self.class_names):
            results_df[f'prob_{class_name}'] = y_prob[:, i]
        
        # 保存详细结果
        detailed_results_path = Path(output_dir) / "detailed_predictions.csv"
        results_df.to_csv(detailed_results_path, index=False)
        
        print(f"Detailed results saved to: {detailed_results_path}")
        
        # 分析错误预测
        error_df = results_df[~results_df['correct']].copy()
        if len(error_df) > 0:
            error_analysis_path = Path(output_dir) / "error_analysis.csv"
            error_df.to_csv(error_analysis_path, index=False)
            print(f"Error analysis saved to: {error_analysis_path}")
            print(f"Number of misclassified samples: {len(error_df)}")
    
    def print_results(self, results: Dict[str, Any]):
        """打印评估结果"""
        print("\n" + "="*60)
        print(f"{results['model_name'].upper()} MODEL EVALUATION RESULTS")
        print("="*60)
        
        print(f"Model: {results['model_path']}")
        print(f"Test Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
        print(f"Correct Predictions: {results['correct_predictions']}/{results['total_samples']}")
        print(f"Incorrect Predictions: {results['incorrect_predictions']}")
        
        print(f"\nOverall Metrics:")
        print(f"  Macro Precision: {results['macro_precision']:.4f}")
        print(f"  Macro Recall: {results['macro_recall']:.4f}")
        print(f"  Macro F1-Score: {results['macro_f1']:.4f}")
        print(f"  Micro Precision: {results['micro_precision']:.4f}")
        print(f"  Micro Recall: {results['micro_recall']:.4f}")
        print(f"  Micro F1-Score: {results['micro_f1']:.4f}")
        
        print(f"\nPer-Class Metrics:")
        for class_name, metrics in results['per_class_metrics'].items():
            print(f"  {class_name}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1-Score: {metrics['f1_score']:.4f}")
            print(f"    Support: {metrics['support']}")
        
        print(f"\nResults saved to: {results['output_dir']}")
        print("="*60)


class ModelEvaluator(BaseEvaluator):
    """基线模型评估器"""
    
    def _load_model(self) -> torch.nn.Module:
        """加载基线模型"""
        print(f"Loading baseline model from: {self.model_path}")
        
        # 创建模型
        model = create_resnet_baseline(
            num_classes=config.NUM_CLASSES,
            pretrained=False,
            dropout_rate=0.5,
            freeze_backbone=False
        )
        
        # 加载权重
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 处理不同的保存格式
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            if 'val_accuracy' in checkpoint:
                print(f"Model validation accuracy: {checkpoint['val_accuracy']:.4f}")
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        print(f"Model loaded successfully on {self.device}")
        return model
    
    def _get_model_name(self) -> str:
        return "baseline"


class AttentionModelEvaluator(BaseEvaluator):
    """注意力模型评估器"""
    
    def __init__(self, model_path: str, model_type: str, device: str = None):
        """
        初始化注意力模型评估器
        
        Args:
            model_path: 训练好的模型权重路径
            model_type: 模型类型 ('senet' 或 'cbam')
            device: 计算设备
        """
        self.model_type = model_type.lower()
        if self.model_type not in ['senet', 'cbam']:
            raise ValueError("model_type must be 'senet' or 'cbam'")
        
        super().__init__(model_path, device)
    
    def _load_model(self) -> torch.nn.Module:
        """加载注意力模型"""
        print(f"Loading {self.model_type.upper()} model from: {self.model_path}")
        
        # 创建模型
        if self.model_type == 'senet':
            model = create_senet_model(
                num_classes=config.NUM_CLASSES,
                pretrained=False,
                reduction=16
            )
        else:  # cbam
            model = create_cbam_model(
                num_classes=config.NUM_CLASSES,
                pretrained=False,
                reduction=16
            )
        
        # 加载权重
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 处理不同的保存格式
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            if 'val_accuracy' in checkpoint:
                print(f"Model validation accuracy: {checkpoint['val_accuracy']:.4f}")
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        print(f"{self.model_type.upper()} model loaded successfully on {self.device}")
        return model
    
    def _get_model_name(self) -> str:
        return self.model_type
    
    def evaluate_with_attention(self, save_attention: bool = True) -> Dict[str, Any]:
        """评估模型并提取注意力信息"""
        print(f"Starting {self.model_type.upper()} model evaluation with attention analysis...")
        
        # 基础评估
        results = self.evaluate(save_predictions=True)
        
        if save_attention:
            # 提取注意力信息
            attention_info = self._extract_attention_info()
            results['attention_info'] = attention_info
            
            # 保存注意力分析
            self._save_attention_analysis(attention_info, results['output_dir'])
        
        return results
    
    def _extract_attention_info(self) -> Dict[str, Any]:
        """提取注意力信息"""
        print("Extracting attention information...")
        
        # 加载少量测试数据进行注意力分析
        test_loader = self._load_test_data()
        
        attention_weights = []
        sample_count = 0
        max_samples = 100  # 限制样本数量以节省时间
        
        self.model.eval()
        with torch.no_grad():
            for images, labels, image_paths in test_loader:
                if sample_count >= max_samples:
                    break
                    
                images = images.to(self.device)
                
                # 获取注意力权重
                if self.model_type == 'senet':
                    attention_data = self._extract_senet_attention(images)
                else:  # cbam
                    attention_data = self._extract_cbam_attention(images)
                
                attention_weights.append(attention_data)
                sample_count += len(images)
        
        # 分析注意力权重
        attention_analysis = self._analyze_attention_weights(attention_weights)
        
        return {
            'model_type': self.model_type,
            'sample_count': sample_count,
            'attention_analysis': attention_analysis
        }
    
    def _extract_senet_attention(self, images: torch.Tensor) -> Dict[str, Any]:
        """提取SE-Net注意力权重"""
        attention_weights = {}
        
        # 注册钩子函数来捕获SE模块的输出
        def hook_fn(name):
            def hook(module, input, output):
                if hasattr(module, 'se'):
                    # SE模块的输出就是注意力权重
                    attention_weights[name] = output.detach().cpu()
            return hook
        
        # 注册钩子
        hooks = []
        for name, module in self.model.named_modules():
            if 'se' in name and hasattr(module, 'fc2'):
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # 前向传播
        _ = self.model(images)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        return attention_weights
    
    def _extract_cbam_attention(self, images: torch.Tensor) -> Dict[str, Any]:
        """提取CBAM注意力权重"""
        attention_weights = {}
        
        # 注册钩子函数来捕获CBAM模块的输出
        def hook_fn(name):
            def hook(module, input, output):
                attention_weights[name] = {
                    'channel_attention': getattr(module, '_last_channel_attention', None),
                    'spatial_attention': getattr(module, '_last_spatial_attention', None)
                }
            return hook
        
        # 注册钩子
        hooks = []
        for name, module in self.model.named_modules():
            if 'cbam' in name:
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # 前向传播
        _ = self.model(images)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        return attention_weights
    
    def _analyze_attention_weights(self, attention_weights: List[Dict]) -> Dict[str, Any]:
        """分析注意力权重"""
        # 这里可以添加更详细的注意力分析
        # 例如：权重分布统计、激活模式分析等
        
        analysis = {
            'total_batches': len(attention_weights),
            'attention_layers': [],
            'statistics': {}
        }
        
        if attention_weights:
            # 获取注意力层信息
            first_batch = attention_weights[0]
            analysis['attention_layers'] = list(first_batch.keys())
            
            # 计算基本统计信息
            for layer_name in analysis['attention_layers']:
                layer_stats = {
                    'mean_activation': 0.0,
                    'std_activation': 0.0,
                    'max_activation': 0.0,
                    'min_activation': 0.0
                }
                analysis['statistics'][layer_name] = layer_stats
        
        return analysis
    
    def _save_attention_analysis(self, attention_info: Dict[str, Any], output_dir: str):
        """保存注意力分析结果"""
        attention_path = Path(output_dir) / "attention_analysis.json"
        
        # 转换为可序列化的格式
        serializable_info = {
            'model_type': attention_info['model_type'],
            'sample_count': attention_info['sample_count'],
            'attention_analysis': attention_info['attention_analysis']
        }
        
        with open(attention_path, 'w') as f:
            json.dump(serializable_info, f, indent=2)
        
        print(f"Attention analysis saved to: {attention_path}")


def find_best_checkpoint(model_dir: str) -> str:
    """找到最佳的检查点文件"""
    model_path = Path(model_dir)
    
    # 查找所有best_checkpoint文件
    best_checkpoints = list(model_path.glob("best_checkpoint_*.pth"))
    
    if not best_checkpoints:
        # 如果没有best_checkpoint，查找最后的checkpoint
        checkpoints = list(model_path.glob("checkpoint_epoch_*.pth"))
        if checkpoints:
            # 按epoch数排序，取最后一个
            checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
            return str(checkpoints[-1])
        else:
            raise FileNotFoundError(f"No checkpoint files found in {model_dir}")
    
    # 按epoch数排序，取最后一个（通常是最佳的）
    best_checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
    return str(best_checkpoints[-1])


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Evaluate models on test set')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--model_dir', type=str, 
                       default='outputs/models/resnet50_baseline',
                       help='Directory containing model checkpoints')
    parser.add_argument('--model_type', type=str, default='baseline',
                       choices=['baseline', 'senet', 'cbam'],
                       help='Type of model to evaluate')
    parser.add_argument('--save_attention', action='store_true',
                       help='Save attention analysis for attention models')
    
    args = parser.parse_args()
    
    # 确定模型路径
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = find_best_checkpoint(args.model_dir)
    
    print(f"Using model: {model_path}")
    print(f"Model type: {args.model_type}")
    
    # 创建评估器
    if args.model_type == 'baseline':
        evaluator = ModelEvaluator(model_path)
        results = evaluator.evaluate()
    else:
        evaluator = AttentionModelEvaluator(model_path, args.model_type)
        if args.save_attention:
            results = evaluator.evaluate_with_attention()
        else:
            results = evaluator.evaluate()
    
    # 打印结果
    evaluator.print_results(results)
    
    # 保存结果摘要
    results_summary_path = Path(results['output_dir']) / "evaluation_summary.json"
    with open(results_summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nEvaluation summary saved to: {results_summary_path}")


if __name__ == "__main__":
    main() 