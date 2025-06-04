#!/usr/bin/env python3
"""
简化的基线模型评估脚本
直接运行基线模型的测试集评估
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.evaluation.evaluate import ModelEvaluator, find_best_checkpoint


def main():
    """主函数"""
    print("Starting baseline model evaluation...")
    
    # 自动找到最佳模型
    model_dir = "outputs/models/resnet50_baseline"
    try:
        model_path = find_best_checkpoint(model_dir)
        print(f"Found best model: {model_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # 创建评估器并运行评估
    evaluator = ModelEvaluator(model_path)
    results = evaluator.evaluate()
    
    # 打印结果
    evaluator.print_results(results)
    
    print("\n✅ Baseline model evaluation completed successfully!")


if __name__ == "__main__":
    main() 