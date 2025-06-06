#!/usr/bin/env python3
"""
模型比较器模块

提供多个模型之间的性能比较和分析功能
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ModelResult:
    """模型结果数据类"""
    name: str
    accuracy: float
    macro_f1: float
    micro_f1: float
    per_class_f1: Dict[str, float]
    confusion_matrix: np.ndarray
    total_params: Optional[int] = None
    training_time: Optional[float] = None
    inference_time: Optional[float] = None
    model_size: Optional[float] = None  # MB
    
    @classmethod
    def from_evaluation_result(cls, result: Dict[str, Any], name: str = None) -> 'ModelResult':
        """从评估结果创建ModelResult对象"""
        model_name = name or result.get('model_name', 'unknown')
        
        # 提取每个类别的F1分数
        per_class_f1 = {}
        if 'per_class_metrics' in result:
            for class_name, metrics in result['per_class_metrics'].items():
                per_class_f1[class_name] = metrics['f1_score']
        
        # 提取混淆矩阵
        confusion_matrix = np.array([])
        if 'classification_report' in result:
            # 从sklearn的classification_report中提取混淆矩阵信息
            pass
        
        return cls(
            name=model_name,
            accuracy=result.get('test_accuracy', 0.0),
            macro_f1=result.get('macro_f1', 0.0),
            micro_f1=result.get('micro_f1', 0.0),
            per_class_f1=per_class_f1,
            confusion_matrix=confusion_matrix
        )


class ModelComparator:
    """模型比较器"""
    
    def __init__(self, class_names: List[str]):
        """
        初始化模型比较器
        
        Args:
            class_names: 类别名称列表
        """
        self.class_names = class_names
        self.models: List[ModelResult] = []
    
    def add_model(self, model_result: ModelResult):
        """添加模型结果"""
        self.models.append(model_result)
    
    def add_model_from_file(self, result_file: str, model_name: str = None):
        """从结果文件添加模型"""
        with open(result_file, 'r') as f:
            result = json.load(f)
        
        model_result = ModelResult.from_evaluation_result(result, model_name)
        self.add_model(model_result)
    
    def add_model_from_dict(self, result_dict: Dict[str, Any], model_name: str = None):
        """从结果字典添加模型"""
        model_result = ModelResult.from_evaluation_result(result_dict, model_name)
        self.add_model(model_result)
    
    def compare_overall_performance(self) -> pd.DataFrame:
        """比较整体性能"""
        if not self.models:
            return pd.DataFrame()
        
        data = []
        for model in self.models:
            row = {
                'Model': model.name,
                'Accuracy': model.accuracy,
                'Macro F1': model.macro_f1,
                'Micro F1': model.micro_f1
            }
            
            # 添加可选指标
            if model.total_params is not None:
                row['Parameters (M)'] = model.total_params / 1e6
            if model.training_time is not None:
                row['Training Time (min)'] = model.training_time / 60
            if model.inference_time is not None:
                row['Inference Time (ms)'] = model.inference_time * 1000
            if model.model_size is not None:
                row['Model Size (MB)'] = model.model_size
            
            data.append(row)
        
        df = pd.DataFrame(data)
        return df.sort_values('Accuracy', ascending=False)
    
    def compare_per_class_performance(self) -> pd.DataFrame:
        """比较每个类别的性能"""
        if not self.models:
            return pd.DataFrame()
        
        data = []
        for model in self.models:
            for class_name in self.class_names:
                f1_score = model.per_class_f1.get(class_name, 0.0)
                data.append({
                    'Model': model.name,
                    'Class': class_name,
                    'F1 Score': f1_score
                })
        
        return pd.DataFrame(data)
    
    def calculate_performance_ranking(self) -> Dict[str, List[str]]:
        """计算性能排名"""
        if not self.models:
            return {}
        
        rankings = {}
        
        # 按准确率排名
        accuracy_ranking = sorted(self.models, key=lambda x: x.accuracy, reverse=True)
        rankings['accuracy'] = [model.name for model in accuracy_ranking]
        
        # 按宏F1排名
        macro_f1_ranking = sorted(self.models, key=lambda x: x.macro_f1, reverse=True)
        rankings['macro_f1'] = [model.name for model in macro_f1_ranking]
        
        # 按微F1排名
        micro_f1_ranking = sorted(self.models, key=lambda x: x.micro_f1, reverse=True)
        rankings['micro_f1'] = [model.name for model in micro_f1_ranking]
        
        # 综合排名（平均排名）
        model_scores = {}
        for model in self.models:
            acc_rank = rankings['accuracy'].index(model.name)
            macro_rank = rankings['macro_f1'].index(model.name)
            micro_rank = rankings['micro_f1'].index(model.name)
            avg_rank = (acc_rank + macro_rank + micro_rank) / 3
            model_scores[model.name] = avg_rank
        
        overall_ranking = sorted(model_scores.items(), key=lambda x: x[1])
        rankings['overall'] = [model_name for model_name, _ in overall_ranking]
        
        return rankings
    
    def analyze_performance_differences(self) -> Dict[str, Any]:
        """分析性能差异"""
        if len(self.models) < 2:
            return {}
        
        # 计算统计信息
        accuracies = [model.accuracy for model in self.models]
        macro_f1s = [model.macro_f1 for model in self.models]
        micro_f1s = [model.micro_f1 for model in self.models]
        
        analysis = {
            'accuracy_stats': {
                'mean': float(np.mean(accuracies)),
                'std': float(np.std(accuracies)),
                'min': float(np.min(accuracies)),
                'max': float(np.max(accuracies)),
                'range': float(np.max(accuracies) - np.min(accuracies))
            },
            'macro_f1_stats': {
                'mean': float(np.mean(macro_f1s)),
                'std': float(np.std(macro_f1s)),
                'min': float(np.min(macro_f1s)),
                'max': float(np.max(macro_f1s)),
                'range': float(np.max(macro_f1s) - np.min(macro_f1s))
            },
            'micro_f1_stats': {
                'mean': float(np.mean(micro_f1s)),
                'std': float(np.std(micro_f1s)),
                'min': float(np.min(micro_f1s)),
                'max': float(np.max(micro_f1s)),
                'range': float(np.max(micro_f1s) - np.min(micro_f1s))
            }
        }
        
        # 找出最佳和最差模型
        best_accuracy_idx = np.argmax(accuracies)
        worst_accuracy_idx = np.argmin(accuracies)
        
        analysis['best_model'] = {
            'name': self.models[best_accuracy_idx].name,
            'accuracy': self.models[best_accuracy_idx].accuracy,
            'macro_f1': self.models[best_accuracy_idx].macro_f1
        }
        
        analysis['worst_model'] = {
            'name': self.models[worst_accuracy_idx].name,
            'accuracy': self.models[worst_accuracy_idx].accuracy,
            'macro_f1': self.models[worst_accuracy_idx].macro_f1
        }
        
        # 计算性能提升
        analysis['performance_improvement'] = {
            'accuracy': float(analysis['best_model']['accuracy'] - analysis['worst_model']['accuracy']),
            'macro_f1': float(analysis['best_model']['macro_f1'] - analysis['worst_model']['macro_f1'])
        }
        
        return analysis
    
    def generate_comparison_report(self) -> Dict[str, Any]:
        """生成比较报告"""
        if not self.models:
            return {'error': 'No models to compare'}
        
        report = {
            'summary': {
                'total_models': len(self.models),
                'model_names': [model.name for model in self.models],
                'comparison_date': pd.Timestamp.now().isoformat()
            },
            'overall_performance': self.compare_overall_performance().to_dict('records'),
            'per_class_performance': self.compare_per_class_performance().to_dict('records'),
            'rankings': self.calculate_performance_ranking(),
            'performance_analysis': self.analyze_performance_differences()
        }
        
        return report
    
    def plot_comparison_charts(self, output_dir: str):
        """绘制比较图表"""
        if not self.models:
            print("No models to compare")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 整体性能比较条形图
        self._plot_overall_performance_comparison(output_path)
        
        # 2. 每个类别的F1分数比较
        self._plot_per_class_f1_comparison(output_path)
        
        # 3. 性能雷达图
        self._plot_performance_radar(output_path)
        
        # 4. 性能分布箱线图
        self._plot_performance_distribution(output_path)
        
        print(f"Comparison charts saved to: {output_path}")
    
    def _plot_overall_performance_comparison(self, output_path: Path):
        """绘制整体性能比较图"""
        df = self.compare_overall_performance()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 准确率比较
        axes[0].bar(df['Model'], df['Accuracy'], color='skyblue', alpha=0.7)
        axes[0].set_title('Accuracy Comparison')
        axes[0].set_ylabel('Accuracy')
        axes[0].tick_params(axis='x', rotation=45)
        for i, v in enumerate(df['Accuracy']):
            axes[0].text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom')
        
        # 宏F1比较
        axes[1].bar(df['Model'], df['Macro F1'], color='lightgreen', alpha=0.7)
        axes[1].set_title('Macro F1 Comparison')
        axes[1].set_ylabel('Macro F1')
        axes[1].tick_params(axis='x', rotation=45)
        for i, v in enumerate(df['Macro F1']):
            axes[1].text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom')
        
        # 微F1比较
        axes[2].bar(df['Model'], df['Micro F1'], color='salmon', alpha=0.7)
        axes[2].set_title('Micro F1 Comparison')
        axes[2].set_ylabel('Micro F1')
        axes[2].tick_params(axis='x', rotation=45)
        for i, v in enumerate(df['Micro F1']):
            axes[2].text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path / 'overall_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_per_class_f1_comparison(self, output_path: Path):
        """绘制每个类别的F1分数比较"""
        df = self.compare_per_class_performance()
        
        plt.figure(figsize=(12, 8))
        
        # 创建分组条形图
        pivot_df = df.pivot(index='Class', columns='Model', values='F1 Score')
        
        ax = pivot_df.plot(kind='bar', figsize=(12, 8), alpha=0.7)
        plt.title('F1 Score Comparison by Class')
        plt.xlabel('Class')
        plt.ylabel('F1 Score')
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'per_class_f1_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_radar(self, output_path: Path):
        """绘制性能雷达图"""
        if len(self.models) > 5:  # 限制模型数量以保持图表清晰
            models_to_plot = self.models[:5]
        else:
            models_to_plot = self.models
        
        # 准备数据
        categories = ['Accuracy', 'Macro F1', 'Micro F1']
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models_to_plot)))
        
        for i, model in enumerate(models_to_plot):
            values = [model.accuracy, model.macro_f1, model.micro_f1]
            values += values[:1]  # 闭合图形
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model.name, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar Chart', size=16, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path / 'performance_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_distribution(self, output_path: Path):
        """绘制性能分布箱线图"""
        # 准备数据
        data = []
        for model in self.models:
            for class_name in self.class_names:
                f1_score = model.per_class_f1.get(class_name, 0.0)
                data.append({
                    'Model': model.name,
                    'F1 Score': f1_score
                })
        
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df, x='Model', y='F1 Score')
        plt.title('F1 Score Distribution by Model')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'performance_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_comparison_report(self, output_file: str):
        """保存比较报告"""
        report = self.generate_comparison_report()
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Comparison report saved to: {output_file}")
    
    def print_comparison_summary(self):
        """打印比较摘要"""
        if not self.models:
            print("No models to compare")
            return
        
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        
        # 整体性能表格
        df = self.compare_overall_performance()
        print("\nOverall Performance:")
        print(df.to_string(index=False, float_format='%.4f'))
        
        # 排名信息
        rankings = self.calculate_performance_ranking()
        print(f"\nPerformance Rankings:")
        print(f"  Best Accuracy: {rankings['accuracy'][0]}")
        print(f"  Best Macro F1: {rankings['macro_f1'][0]}")
        print(f"  Best Overall: {rankings['overall'][0]}")
        
        # 性能分析
        analysis = self.analyze_performance_differences()
        if analysis:
            print(f"\nPerformance Analysis:")
            print(f"  Best Model: {analysis['best_model']['name']} (Acc: {analysis['best_model']['accuracy']:.4f})")
            print(f"  Worst Model: {analysis['worst_model']['name']} (Acc: {analysis['worst_model']['accuracy']:.4f})")
            print(f"  Accuracy Improvement: {analysis['performance_improvement']['accuracy']:.4f}")
            print(f"  Macro F1 Improvement: {analysis['performance_improvement']['macro_f1']:.4f}")
        
        print("="*60) 