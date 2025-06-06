#!/usr/bin/env python3
"""
指标计算和性能分析模块

提供各种评估指标的计算和性能分析功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class MetricsCalculator:
    """指标计算器"""
    
    def __init__(self, class_names: List[str]):
        """
        初始化指标计算器
        
        Args:
            class_names: 类别名称列表
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算基础分类指标"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'macro_precision': precision_score(y_true, y_pred, average='macro'),
            'macro_recall': recall_score(y_true, y_pred, average='macro'),
            'macro_f1': f1_score(y_true, y_pred, average='macro'),
            'micro_precision': precision_score(y_true, y_pred, average='micro'),
            'micro_recall': recall_score(y_true, y_pred, average='micro'),
            'micro_f1': f1_score(y_true, y_pred, average='micro'),
            'weighted_precision': precision_score(y_true, y_pred, average='weighted'),
            'weighted_recall': recall_score(y_true, y_pred, average='weighted'),
            'weighted_f1': f1_score(y_true, y_pred, average='weighted')
        }
        return metrics
    
    def calculate_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """计算每个类别的详细指标"""
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(self.num_classes)
        )
        
        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            per_class_metrics[class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
        
        return per_class_metrics
    
    def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """计算混淆矩阵相关指标"""
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        return {
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_normalized': cm_normalized.tolist(),
            'true_positives': np.diag(cm).tolist(),
            'false_positives': (cm.sum(axis=0) - np.diag(cm)).tolist(),
            'false_negatives': (cm.sum(axis=1) - np.diag(cm)).tolist(),
            'true_negatives': (cm.sum() - cm.sum(axis=0) - cm.sum(axis=1) + np.diag(cm)).tolist()
        }
    
    def calculate_roc_metrics(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """计算ROC相关指标（多分类）"""
        from sklearn.preprocessing import label_binarize
        
        # 二值化标签
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
        
        # 计算每个类别的ROC AUC
        roc_auc_per_class = {}
        fpr_per_class = {}
        tpr_per_class = {}
        
        for i, class_name in enumerate(self.class_names):
            if self.num_classes == 2:
                # 二分类情况
                fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                roc_auc = roc_auc_score(y_true, y_prob[:, 1])
            else:
                # 多分类情况
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
            
            roc_auc_per_class[class_name] = float(roc_auc)
            fpr_per_class[class_name] = fpr.tolist()
            tpr_per_class[class_name] = tpr.tolist()
        
        # 计算宏平均和微平均ROC AUC
        if self.num_classes > 2:
            macro_roc_auc = roc_auc_score(y_true_bin, y_prob, average='macro', multi_class='ovr')
            micro_roc_auc = roc_auc_score(y_true_bin, y_prob, average='micro', multi_class='ovr')
        else:
            macro_roc_auc = roc_auc_per_class[self.class_names[1]]
            micro_roc_auc = macro_roc_auc
        
        return {
            'roc_auc_per_class': roc_auc_per_class,
            'macro_roc_auc': float(macro_roc_auc),
            'micro_roc_auc': float(micro_roc_auc),
            'fpr_per_class': fpr_per_class,
            'tpr_per_class': tpr_per_class
        }
    
    def calculate_precision_recall_metrics(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """计算Precision-Recall相关指标"""
        from sklearn.preprocessing import label_binarize
        
        # 二值化标签
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
        
        # 计算每个类别的PR指标
        pr_auc_per_class = {}
        precision_per_class = {}
        recall_per_class = {}
        
        for i, class_name in enumerate(self.class_names):
            if self.num_classes == 2:
                precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
                pr_auc = average_precision_score(y_true, y_prob[:, 1])
            else:
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
                pr_auc = average_precision_score(y_true_bin[:, i], y_prob[:, i])
            
            pr_auc_per_class[class_name] = float(pr_auc)
            precision_per_class[class_name] = precision.tolist()
            recall_per_class[class_name] = recall.tolist()
        
        # 计算宏平均和微平均PR AUC
        if self.num_classes > 2:
            macro_pr_auc = average_precision_score(y_true_bin, y_prob, average='macro')
            micro_pr_auc = average_precision_score(y_true_bin, y_prob, average='micro')
        else:
            macro_pr_auc = pr_auc_per_class[self.class_names[1]]
            micro_pr_auc = macro_pr_auc
        
        return {
            'pr_auc_per_class': pr_auc_per_class,
            'macro_pr_auc': float(macro_pr_auc),
            'micro_pr_auc': float(micro_pr_auc),
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class
        }
    
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """计算所有指标"""
        metrics = {
            'basic_metrics': self.calculate_basic_metrics(y_true, y_pred),
            'per_class_metrics': self.calculate_per_class_metrics(y_true, y_pred),
            'confusion_matrix_metrics': self.calculate_confusion_matrix(y_true, y_pred)
        }
        
        if y_prob is not None:
            metrics['roc_metrics'] = self.calculate_roc_metrics(y_true, y_prob)
            metrics['pr_metrics'] = self.calculate_precision_recall_metrics(y_true, y_prob)
        
        return metrics


class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self, class_names: List[str]):
        """
        初始化性能分析器
        
        Args:
            class_names: 类别名称列表
        """
        self.class_names = class_names
        self.metrics_calculator = MetricsCalculator(class_names)
    
    def analyze_errors(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_prob: np.ndarray, image_paths: List[str]) -> Dict[str, Any]:
        """分析错误预测"""
        # 找出错误预测的样本
        error_mask = y_true != y_pred
        error_indices = np.where(error_mask)[0]
        
        if len(error_indices) == 0:
            return {
                'total_errors': 0,
                'error_rate': 0.0,
                'error_by_class': {},
                'most_confused_pairs': [],
                'low_confidence_errors': [],
                'high_confidence_errors': []
            }
        
        # 错误样本信息
        error_true_labels = y_true[error_indices]
        error_pred_labels = y_pred[error_indices]
        error_probs = y_prob[error_indices]
        error_paths = [image_paths[i] for i in error_indices]
        error_confidences = np.max(error_probs, axis=1)
        
        # 按类别统计错误
        error_by_class = {}
        for i, class_name in enumerate(self.class_names):
            class_errors = np.sum(error_true_labels == i)
            class_total = np.sum(y_true == i)
            error_by_class[class_name] = {
                'errors': int(class_errors),
                'total': int(class_total),
                'error_rate': float(class_errors / class_total) if class_total > 0 else 0.0
            }
        
        # 找出最容易混淆的类别对
        confusion_pairs = {}
        for true_idx, pred_idx in zip(error_true_labels, error_pred_labels):
            pair = (self.class_names[true_idx], self.class_names[pred_idx])
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
        
        most_confused_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
        
        # 分析置信度
        confidence_threshold = 0.8
        low_confidence_errors = []
        high_confidence_errors = []
        
        for i, (true_label, pred_label, confidence, path) in enumerate(
            zip(error_true_labels, error_pred_labels, error_confidences, error_paths)
        ):
            error_info = {
                'index': int(error_indices[i]),
                'image_path': path,
                'true_label': self.class_names[true_label],
                'predicted_label': self.class_names[pred_label],
                'confidence': float(confidence)
            }
            
            if confidence < confidence_threshold:
                low_confidence_errors.append(error_info)
            else:
                high_confidence_errors.append(error_info)
        
        return {
            'total_errors': len(error_indices),
            'error_rate': float(len(error_indices) / len(y_true)),
            'error_by_class': error_by_class,
            'most_confused_pairs': most_confused_pairs[:10],  # 前10个最混淆的对
            'low_confidence_errors': sorted(low_confidence_errors, key=lambda x: x['confidence']),
            'high_confidence_errors': sorted(high_confidence_errors, key=lambda x: x['confidence'], reverse=True)
        }
    
    def analyze_confidence_distribution(self, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """分析预测置信度分布"""
        confidences = np.max(y_prob, axis=1)
        
        # 基本统计
        confidence_stats = {
            'mean': float(np.mean(confidences)),
            'std': float(np.std(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences)),
            'median': float(np.median(confidences)),
            'q25': float(np.percentile(confidences, 25)),
            'q75': float(np.percentile(confidences, 75))
        }
        
        # 按置信度区间统计
        confidence_bins = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
        confidence_distribution = {}
        
        for i in range(len(confidence_bins) - 1):
            low, high = confidence_bins[i], confidence_bins[i + 1]
            mask = (confidences >= low) & (confidences < high) if i < len(confidence_bins) - 2 else (confidences >= low) & (confidences <= high)
            count = np.sum(mask)
            confidence_distribution[f'{low:.2f}-{high:.2f}'] = {
                'count': int(count),
                'percentage': float(count / len(confidences) * 100)
            }
        
        # 按类别分析置信度
        confidence_by_class = {}
        for i, class_name in enumerate(self.class_names):
            class_mask = y_pred == i
            if np.sum(class_mask) > 0:
                class_confidences = confidences[class_mask]
                confidence_by_class[class_name] = {
                    'mean': float(np.mean(class_confidences)),
                    'std': float(np.std(class_confidences)),
                    'count': int(np.sum(class_mask))
                }
        
        return {
            'overall_stats': confidence_stats,
            'distribution': confidence_distribution,
            'by_class': confidence_by_class
        }
    
    def generate_performance_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  y_prob: np.ndarray, image_paths: List[str]) -> Dict[str, Any]:
        """生成完整的性能报告"""
        # 计算所有指标
        all_metrics = self.metrics_calculator.calculate_all_metrics(y_true, y_pred, y_prob)
        
        # 错误分析
        error_analysis = self.analyze_errors(y_true, y_pred, y_prob, image_paths)
        
        # 置信度分析
        confidence_analysis = self.analyze_confidence_distribution(y_prob, y_pred)
        
        # 组合报告
        report = {
            'summary': {
                'total_samples': len(y_true),
                'correct_predictions': int(np.sum(y_true == y_pred)),
                'accuracy': all_metrics['basic_metrics']['accuracy'],
                'macro_f1': all_metrics['basic_metrics']['macro_f1'],
                'total_errors': error_analysis['total_errors'],
                'error_rate': error_analysis['error_rate']
            },
            'metrics': all_metrics,
            'error_analysis': error_analysis,
            'confidence_analysis': confidence_analysis,
            'class_names': self.class_names
        }
        
        return report
    
    def plot_performance_charts(self, report: Dict[str, Any], output_dir: str):
        """绘制性能分析图表"""
        output_path = Path(output_dir)
        
        # 1. 每个类别的F1分数条形图
        self._plot_per_class_f1(report, output_path)
        
        # 2. 置信度分布直方图
        self._plot_confidence_distribution(report, output_path)
        
        # 3. 错误率按类别分析
        self._plot_error_rate_by_class(report, output_path)
        
        # 4. ROC曲线（如果有ROC指标）
        if 'roc_metrics' in report['metrics']:
            self._plot_roc_curves(report, output_path)
        
        # 5. Precision-Recall曲线（如果有PR指标）
        if 'pr_metrics' in report['metrics']:
            self._plot_pr_curves(report, output_path)
    
    def _plot_per_class_f1(self, report: Dict[str, Any], output_path: Path):
        """绘制每个类别的F1分数"""
        per_class_metrics = report['metrics']['per_class_metrics']
        
        classes = list(per_class_metrics.keys())
        f1_scores = [per_class_metrics[cls]['f1_score'] for cls in classes]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(classes, f1_scores, color='skyblue', alpha=0.7)
        plt.title('F1 Score by Class')
        plt.xlabel('Class')
        plt.ylabel('F1 Score')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # 添加数值标签
        for bar, score in zip(bars, f1_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path / 'f1_scores_by_class.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_distribution(self, report: Dict[str, Any], output_path: Path):
        """绘制置信度分布"""
        confidence_dist = report['confidence_analysis']['distribution']
        
        bins = list(confidence_dist.keys())
        counts = [confidence_dist[bin_name]['count'] for bin_name in bins]
        
        plt.figure(figsize=(10, 6))
        plt.bar(bins, counts, color='lightgreen', alpha=0.7)
        plt.title('Prediction Confidence Distribution')
        plt.xlabel('Confidence Range')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_error_rate_by_class(self, report: Dict[str, Any], output_path: Path):
        """绘制各类别错误率"""
        error_by_class = report['error_analysis']['error_by_class']
        
        classes = list(error_by_class.keys())
        error_rates = [error_by_class[cls]['error_rate'] for cls in classes]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(classes, error_rates, color='salmon', alpha=0.7)
        plt.title('Error Rate by Class')
        plt.xlabel('Class')
        plt.ylabel('Error Rate')
        plt.xticks(rotation=45)
        plt.ylim(0, max(error_rates) * 1.1 if error_rates else 1)
        
        # 添加数值标签
        for bar, rate in zip(bars, error_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{rate:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path / 'error_rate_by_class.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curves(self, report: Dict[str, Any], output_path: Path):
        """绘制ROC曲线"""
        roc_metrics = report['metrics']['roc_metrics']
        
        plt.figure(figsize=(10, 8))
        
        for class_name in self.class_names:
            fpr = roc_metrics['fpr_per_class'][class_name]
            tpr = roc_metrics['tpr_per_class'][class_name]
            auc = roc_metrics['roc_auc_per_class'][class_name]
            
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pr_curves(self, report: Dict[str, Any], output_path: Path):
        """绘制Precision-Recall曲线"""
        pr_metrics = report['metrics']['pr_metrics']
        
        plt.figure(figsize=(10, 8))
        
        for class_name in self.class_names:
            precision = pr_metrics['precision_per_class'][class_name]
            recall = pr_metrics['recall_per_class'][class_name]
            auc = pr_metrics['pr_auc_per_class'][class_name]
            
            plt.plot(recall, precision, label=f'{class_name} (AUC = {auc:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'pr_curves.png', dpi=300, bbox_inches='tight')
        plt.close()


# 修复导入错误
from sklearn.metrics import precision_recall_fscore_support 