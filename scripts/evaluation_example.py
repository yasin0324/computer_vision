#!/usr/bin/env python3
"""
评估模块使用示例

演示如何使用新的评估模块进行模型评估
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.evaluation import (
    ModelEvaluator, 
    AttentionModelEvaluator,
    MetricsCalculator,
    PerformanceAnalyzer,
    ModelComparator,
    EvaluationReporter
)
from src.config.config import config


def example_basic_evaluation():
    """基础评估示例"""
    print("="*60)
    print("基础模型评估示例")
    print("="*60)
    
    # 假设的模型路径
    model_path = "outputs/models/resnet50_baseline/best_checkpoint_42.pth"
    
    try:
        # 创建评估器
        evaluator = ModelEvaluator(model_path)
        
        # 运行评估
        results = evaluator.evaluate()
        
        # 打印结果
        evaluator.print_results(results)
        
        print(f"评估结果已保存到: {results['output_dir']}")
        
    except Exception as e:
        print(f"评估失败: {e}")
        print("请确保模型文件存在且路径正确")


def example_attention_evaluation():
    """注意力模型评估示例"""
    print("\n" + "="*60)
    print("注意力模型评估示例")
    print("="*60)
    
    # 假设的SE-Net模型路径
    senet_path = "outputs/models/senet_model/best_checkpoint_28.pth"
    
    try:
        # 创建注意力模型评估器
        evaluator = AttentionModelEvaluator(senet_path, 'senet')
        
        # 运行评估（包含注意力分析）
        results = evaluator.evaluate_with_attention(save_attention=True)
        
        # 打印结果
        evaluator.print_results(results)
        
        print(f"评估结果已保存到: {results['output_dir']}")
        
    except Exception as e:
        print(f"注意力模型评估失败: {e}")
        print("请确保SE-Net模型文件存在且路径正确")


def example_model_comparison():
    """模型比较示例"""
    print("\n" + "="*60)
    print("模型比较示例")
    print("="*60)
    
    # 模拟的评估结果
    baseline_result = {
        'model_name': 'baseline',
        'test_accuracy': 0.9935,
        'macro_f1': 0.9934,
        'micro_f1': 0.9935,
        'per_class_metrics': {
            'bacterial_spot': {'f1_score': 0.9985},
            'septoria_leaf_spot': {'f1_score': 0.9985},
            'target_spot': {'f1_score': 1.0000},
            'healthy': {'f1_score': 1.0000}
        }
    }
    
    senet_result = {
        'model_name': 'senet',
        'test_accuracy': 0.9993,
        'macro_f1': 0.9993,
        'micro_f1': 0.9993,
        'per_class_metrics': {
            'bacterial_spot': {'f1_score': 0.9985},
            'septoria_leaf_spot': {'f1_score': 0.9985},
            'target_spot': {'f1_score': 1.0000},
            'healthy': {'f1_score': 1.0000}
        }
    }
    
    cbam_result = {
        'model_name': 'cbam',
        'test_accuracy': 0.9956,
        'macro_f1': 0.9956,
        'micro_f1': 0.9956,
        'per_class_metrics': {
            'bacterial_spot': {'f1_score': 0.9942},
            'septoria_leaf_spot': {'f1_score': 0.9971},
            'target_spot': {'f1_score': 0.9971},
            'healthy': {'f1_score': 0.9942}
        }
    }
    
    try:
        # 创建比较器
        class_names = list(config.TARGET_CLASSES.values())
        comparator = ModelComparator(class_names)
        
        # 添加模型结果
        comparator.add_model_from_dict(baseline_result)
        comparator.add_model_from_dict(senet_result)
        comparator.add_model_from_dict(cbam_result)
        
        # 打印比较摘要
        comparator.print_comparison_summary()
        
        # 生成比较报告
        comparison_report = comparator.generate_comparison_report()
        
        # 保存报告
        output_dir = Path(project_root) / "outputs" / "evaluation" / "example_comparison"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成图表
        comparator.plot_comparison_charts(str(output_dir))
        
        # 保存JSON报告
        report_file = output_dir / "comparison_report.json"
        comparator.save_comparison_report(str(report_file))
        
        print(f"\n比较结果已保存到: {output_dir}")
        
    except Exception as e:
        print(f"模型比较失败: {e}")


def example_metrics_calculation():
    """指标计算示例"""
    print("\n" + "="*60)
    print("指标计算示例")
    print("="*60)
    
    # 模拟的预测数据
    import numpy as np
    
    # 假设有100个样本，4个类别
    np.random.seed(42)
    y_true = np.random.randint(0, 4, 100)
    y_pred = y_true.copy()
    # 添加一些错误预测
    error_indices = np.random.choice(100, 5, replace=False)
    y_pred[error_indices] = np.random.randint(0, 4, 5)
    
    # 生成概率
    y_prob = np.random.rand(100, 4)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # 归一化
    
    try:
        # 创建指标计算器
        class_names = list(config.TARGET_CLASSES.values())
        calculator = MetricsCalculator(class_names)
        
        # 计算所有指标
        metrics = calculator.calculate_all_metrics(y_true, y_pred, y_prob)
        
        print("基础指标:")
        for metric, value in metrics['basic_metrics'].items():
            print(f"  {metric}: {value:.4f}")
        
        print("\n每个类别的指标:")
        for class_name, class_metrics in metrics['per_class_metrics'].items():
            print(f"  {class_name}:")
            for metric, value in class_metrics.items():
                if metric != 'support':
                    print(f"    {metric}: {value:.4f}")
                else:
                    print(f"    {metric}: {value}")
        
        print("\nROC AUC指标:")
        if 'roc_metrics' in metrics:
            for class_name, auc in metrics['roc_metrics']['roc_auc_per_class'].items():
                print(f"  {class_name}: {auc:.4f}")
        
    except Exception as e:
        print(f"指标计算失败: {e}")


def example_performance_analysis():
    """性能分析示例"""
    print("\n" + "="*60)
    print("性能分析示例")
    print("="*60)
    
    # 模拟的预测数据
    import numpy as np
    
    np.random.seed(42)
    y_true = np.random.randint(0, 4, 100)
    y_pred = y_true.copy()
    # 添加一些错误预测
    error_indices = np.random.choice(100, 8, replace=False)
    y_pred[error_indices] = np.random.randint(0, 4, 8)
    
    # 生成概率和图像路径
    y_prob = np.random.rand(100, 4)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    image_paths = [f"image_{i:03d}.jpg" for i in range(100)]
    
    try:
        # 创建性能分析器
        class_names = list(config.TARGET_CLASSES.values())
        analyzer = PerformanceAnalyzer(class_names)
        
        # 生成完整的性能报告
        report = analyzer.generate_performance_report(y_true, y_pred, y_prob, image_paths)
        
        print("性能摘要:")
        summary = report['summary']
        print(f"  总样本数: {summary['total_samples']}")
        print(f"  正确预测: {summary['correct_predictions']}")
        print(f"  准确率: {summary['accuracy']:.4f}")
        print(f"  宏F1: {summary['macro_f1']:.4f}")
        print(f"  总错误数: {summary['total_errors']}")
        print(f"  错误率: {summary['error_rate']:.4f}")
        
        print("\n错误分析:")
        error_analysis = report['error_analysis']
        print(f"  总错误数: {error_analysis['total_errors']}")
        
        if error_analysis['most_confused_pairs']:
            print("  最容易混淆的类别对:")
            for (true_class, pred_class), count in error_analysis['most_confused_pairs'][:3]:
                print(f"    {true_class} -> {pred_class}: {count}次")
        
        print("\n置信度分析:")
        conf_analysis = report['confidence_analysis']
        stats = conf_analysis['overall_stats']
        print(f"  平均置信度: {stats['mean']:.4f}")
        print(f"  置信度标准差: {stats['std']:.4f}")
        print(f"  最低置信度: {stats['min']:.4f}")
        print(f"  最高置信度: {stats['max']:.4f}")
        
        # 生成图表
        output_dir = Path(project_root) / "outputs" / "evaluation" / "example_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        analyzer.plot_performance_charts(report, str(output_dir))
        print(f"\n性能分析图表已保存到: {output_dir}")
        
    except Exception as e:
        print(f"性能分析失败: {e}")


def example_report_generation():
    """报告生成示例"""
    print("\n" + "="*60)
    print("报告生成示例")
    print("="*60)
    
    # 模拟的评估结果
    evaluation_result = {
        'model_name': 'example_model',
        'model_path': 'outputs/models/example/model.pth',
        'test_accuracy': 0.9935,
        'macro_precision': 0.9934,
        'macro_recall': 0.9934,
        'macro_f1': 0.9934,
        'micro_precision': 0.9935,
        'micro_recall': 0.9935,
        'micro_f1': 0.9935,
        'total_samples': 1379,
        'correct_predictions': 1370,
        'incorrect_predictions': 9,
        'per_class_metrics': {
            'bacterial_spot': {
                'precision': 1.0000,
                'recall': 0.9971,
                'f1_score': 0.9985,
                'support': 344
            },
            'septoria_leaf_spot': {
                'precision': 0.9971,
                'recall': 1.0000,
                'f1_score': 0.9985,
                'support': 345
            },
            'target_spot': {
                'precision': 1.0000,
                'recall': 1.0000,
                'f1_score': 1.0000,
                'support': 345
            },
            'healthy': {
                'precision': 1.0000,
                'recall': 1.0000,
                'f1_score': 1.0000,
                'support': 345
            }
        }
    }
    
    try:
        # 创建报告生成器
        class_names = list(config.TARGET_CLASSES.values())
        reporter = EvaluationReporter(class_names)
        
        # 创建输出目录
        output_dir = Path(project_root) / "outputs" / "evaluation" / "example_reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成HTML报告
        html_file = output_dir / "example_report.html"
        reporter.generate_html_report(evaluation_result, str(html_file))
        print(f"HTML报告已生成: {html_file}")
        
        # 生成Markdown报告
        md_file = output_dir / "example_report.md"
        reporter.generate_markdown_report(evaluation_result, str(md_file))
        print(f"Markdown报告已生成: {md_file}")
        
    except Exception as e:
        print(f"报告生成失败: {e}")


def main():
    """运行所有示例"""
    print("评估模块使用示例")
    print("="*60)
    
    # 运行各个示例
    try:
        example_metrics_calculation()
        example_performance_analysis()
        example_model_comparison()
        example_report_generation()
        
        print("\n" + "="*60)
        print("所有示例运行完成！")
        print("="*60)
        
        print("\n注意:")
        print("- 基础评估和注意力模型评估需要实际的模型文件")
        print("- 其他示例使用模拟数据演示功能")
        print("- 实际使用时请替换为真实的模型路径和数据")
        
    except Exception as e:
        print(f"运行示例时出错: {e}")


if __name__ == "__main__":
    main() 