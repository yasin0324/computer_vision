#!/usr/bin/env python3
"""
基线模型测试集评估脚本
用于评估训练好的ResNet50基线模型在测试集上的性能
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
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
from src.data.dataset import TomatoSpotDataset
from src.config.config import config


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model_path: str, device: str = None):
        """
        初始化评估器
        
        Args:
            model_path: 训练好的模型权重路径
            device: 计算设备
        """
        self.model_path = model_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self.model = self._load_model()
        
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
        
    def _load_model(self) -> torch.nn.Module:
        """加载训练好的模型"""
        print(f"Loading model from: {self.model_path}")
        
        # 创建模型
        model = create_resnet_baseline(
            num_classes=config.NUM_CLASSES,
            pretrained=False,  # 不需要预训练权重，我们要加载自己的权重
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
    
    def evaluate(self) -> Dict[str, Any]:
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
        with torch.no_grad():
            for batch_idx, (images, labels, image_paths) in enumerate(tqdm(test_loader, desc="Evaluating")):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                logits = self.model(images)
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
        
        # 生成混淆矩阵
        self._plot_confusion_matrix(all_labels, all_predictions, results['output_dir'])
        
        # 保存详细结果
        self._save_detailed_results(
            all_labels, all_predictions, all_probabilities, 
            all_image_paths, results['output_dir']
        )
        
        return results
    
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
        output_dir = Path(project_root) / "outputs" / "evaluation" / f"baseline_evaluation_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 组织结果
        results = {
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
            'output_dir': str(output_dir)
        }
        
        return results
    
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
        print("BASELINE MODEL EVALUATION RESULTS")
        print("="*60)
        
        print(f"Model: {results['model_path']}")
        print(f"Test Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
        
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
    parser = argparse.ArgumentParser(description='Evaluate baseline model on test set')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--model_dir', type=str, 
                       default='outputs/models/resnet50_baseline',
                       help='Directory containing model checkpoints')
    
    args = parser.parse_args()
    
    # 确定模型路径
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = find_best_checkpoint(args.model_dir)
    
    print(f"Using model: {model_path}")
    
    # 创建评估器
    evaluator = ModelEvaluator(model_path)
    
    # 运行评估
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