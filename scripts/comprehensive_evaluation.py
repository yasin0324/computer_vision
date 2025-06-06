#!/usr/bin/env python3
"""
综合评估脚本

使用新的评估模块对模型进行全面评估，包括：
1. 基础模型评估
2. 注意力模型评估
3. 模型性能比较
4. 详细报告生成
"""

import sys
import argparse
from pathlib import Path
import json

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


def evaluate_single_model(model_path: str, model_type: str = 'baseline', 
                         save_attention: bool = False) -> dict:
    """评估单个模型"""
    print(f"\n{'='*60}")
    print(f"评估 {model_type.upper()} 模型")
    print(f"{'='*60}")
    
    if model_type == 'baseline':
        evaluator = ModelEvaluator(model_path)
        results = evaluator.evaluate()
    else:
        evaluator = AttentionModelEvaluator(model_path, model_type)
        if save_attention:
            results = evaluator.evaluate_with_attention()
        else:
            results = evaluator.evaluate()
    
    # 打印结果
    evaluator.print_results(results)
    
    # 生成详细的性能分析
    if 'detailed_predictions.csv' in str(results['output_dir']):
        print("\n生成详细性能分析...")
        analyzer = PerformanceAnalyzer(list(config.TARGET_CLASSES.values()))
        
        # 这里需要重新加载预测数据来进行分析
        # 实际使用时可以从evaluator中获取这些数据
        print("详细性能分析需要预测数据，跳过此步骤")
    
    return results


def compare_models(model_results: list, class_names: list) -> dict:
    """比较多个模型的性能"""
    print(f"\n{'='*60}")
    print("模型性能比较")
    print(f"{'='*60}")
    
    # 创建比较器
    comparator = ModelComparator(class_names)
    
    # 添加模型结果
    for result in model_results:
        comparator.add_model_from_dict(result, result.get('model_name', 'unknown'))
    
    # 生成比较报告
    comparison_report = comparator.generate_comparison_report()
    
    # 打印比较摘要
    comparator.print_comparison_summary()
    
    # 生成比较图表
    output_dir = Path(project_root) / "outputs" / "evaluation" / "model_comparison"
    comparator.plot_comparison_charts(str(output_dir))
    
    # 保存比较报告
    report_file = output_dir / "comparison_report.json"
    comparator.save_comparison_report(str(report_file))
    
    return comparison_report


def generate_reports(evaluation_results: dict, comparison_results: dict = None):
    """生成详细的评估报告"""
    print(f"\n{'='*60}")
    print("生成评估报告")
    print(f"{'='*60}")
    
    class_names = list(config.TARGET_CLASSES.values())
    reporter = EvaluationReporter(class_names)
    
    # 为每个模型生成单独的报告
    for result in evaluation_results:
        model_name = result.get('model_name', 'unknown')
        output_dir = Path(result['output_dir'])
        
        # 生成HTML报告
        html_file = output_dir / f"{model_name}_report.html"
        reporter.generate_html_report(result, str(html_file))
        print(f"HTML报告已生成: {html_file}")
        
        # 生成Markdown报告
        md_file = output_dir / f"{model_name}_report.md"
        reporter.generate_markdown_report(result, str(md_file))
        print(f"Markdown报告已生成: {md_file}")
    
    # 生成比较报告
    if comparison_results:
        comparison_dir = Path(project_root) / "outputs" / "evaluation" / "model_comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # HTML比较报告
        html_comparison_file = comparison_dir / "comparison_report.html"
        reporter.generate_comparison_report(comparison_results, str(html_comparison_file), 'html')
        print(f"HTML比较报告已生成: {html_comparison_file}")
        
        # Markdown比较报告
        md_comparison_file = comparison_dir / "comparison_report.md"
        reporter.generate_comparison_report(comparison_results, str(md_comparison_file), 'markdown')
        print(f"Markdown比较报告已生成: {md_comparison_file}")


def find_model_checkpoints() -> dict:
    """查找可用的模型检查点"""
    models_dir = Path(project_root) / "outputs" / "models"
    available_models = {}
    
    # 查找基线模型
    baseline_dir = models_dir / "resnet50_baseline"
    if baseline_dir.exists():
        baseline_checkpoints = list(baseline_dir.glob("best_checkpoint_*.pth"))
        if baseline_checkpoints:
            # 选择最新的检查点
            latest_checkpoint = max(baseline_checkpoints, key=lambda x: x.stat().st_mtime)
            available_models['baseline'] = str(latest_checkpoint)
    
    # 查找SE-Net模型
    senet_dir = models_dir / "senet_model"
    if senet_dir.exists():
        senet_checkpoints = list(senet_dir.glob("best_checkpoint_*.pth"))
        if senet_checkpoints:
            latest_checkpoint = max(senet_checkpoints, key=lambda x: x.stat().st_mtime)
            available_models['senet'] = str(latest_checkpoint)
    
    # 查找CBAM模型
    cbam_dir = models_dir / "cbam_model"
    if cbam_dir.exists():
        cbam_checkpoints = list(cbam_dir.glob("best_checkpoint_*.pth"))
        if cbam_checkpoints:
            latest_checkpoint = max(cbam_checkpoints, key=lambda x: x.stat().st_mtime)
            available_models['cbam'] = str(latest_checkpoint)
    
    return available_models


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='综合模型评估')
    parser.add_argument('--models', nargs='+', 
                       choices=['baseline', 'senet', 'cbam', 'all'],
                       default=['all'],
                       help='要评估的模型类型')
    parser.add_argument('--save_attention', action='store_true',
                       help='保存注意力分析（仅适用于注意力模型）')
    parser.add_argument('--compare', action='store_true',
                       help='比较多个模型的性能')
    parser.add_argument('--generate_reports', action='store_true',
                       help='生成详细的评估报告')
    parser.add_argument('--baseline_path', type=str,
                       help='基线模型路径')
    parser.add_argument('--senet_path', type=str,
                       help='SE-Net模型路径')
    parser.add_argument('--cbam_path', type=str,
                       help='CBAM模型路径')
    
    args = parser.parse_args()
    
    # 查找可用的模型
    available_models = find_model_checkpoints()
    
    # 如果指定了模型路径，使用指定的路径
    if args.baseline_path:
        available_models['baseline'] = args.baseline_path
    if args.senet_path:
        available_models['senet'] = args.senet_path
    if args.cbam_path:
        available_models['cbam'] = args.cbam_path
    
    print("可用的模型:")
    for model_type, model_path in available_models.items():
        print(f"  {model_type}: {model_path}")
    
    if not available_models:
        print("未找到可用的模型检查点！")
        return
    
    # 确定要评估的模型
    if 'all' in args.models:
        models_to_evaluate = list(available_models.keys())
    else:
        models_to_evaluate = [m for m in args.models if m in available_models]
    
    if not models_to_evaluate:
        print("没有找到要评估的模型！")
        return
    
    print(f"\n将评估以下模型: {models_to_evaluate}")
    
    # 评估每个模型
    evaluation_results = []
    
    for model_type in models_to_evaluate:
        model_path = available_models[model_type]
        
        try:
            result = evaluate_single_model(
                model_path, 
                model_type, 
                save_attention=args.save_attention and model_type != 'baseline'
            )
            evaluation_results.append(result)
            
        except Exception as e:
            print(f"评估 {model_type} 模型时出错: {e}")
            continue
    
    # 模型比较
    comparison_results = None
    if args.compare and len(evaluation_results) > 1:
        try:
            class_names = list(config.TARGET_CLASSES.values())
            comparison_results = compare_models(evaluation_results, class_names)
        except Exception as e:
            print(f"模型比较时出错: {e}")
    
    # 生成报告
    if args.generate_reports:
        try:
            generate_reports(evaluation_results, comparison_results)
        except Exception as e:
            print(f"生成报告时出错: {e}")
    
    print(f"\n{'='*60}")
    print("综合评估完成！")
    print(f"{'='*60}")
    
    # 打印摘要
    print("\n评估摘要:")
    for result in evaluation_results:
        model_name = result.get('model_name', 'unknown')
        accuracy = result.get('test_accuracy', 0)
        print(f"  {model_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")


if __name__ == "__main__":
    main() 