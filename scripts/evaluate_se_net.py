#!/usr/bin/env python3
"""
SE-Net模型评估脚本
"""

import sys
import os
import argparse
from pathlib import Path
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from torchvision import transforms
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.attention_models import create_resnet_with_attention
from src.data.dataset import TomatoSpotDataset
from src.config.config import config


def load_se_net_model(checkpoint_path, device):
    """加载SE-Net模型"""
    # 创建SE-Net模型
    model = create_resnet_with_attention(
        num_classes=config.NUM_CLASSES,
        attention_type='se',
        reduction=16,
        dropout_rate=0.7,
        pretrained=False
    )
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 提取模型状态字典
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 'unknown')
    else:
        state_dict = checkpoint
        epoch = 'unknown'
    
    # 加载权重
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"SE-Net模型加载成功，来自epoch {epoch}")
    return model


def evaluate_model(model, test_loader, device, class_names):
    """评估模型性能"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="评估中"):
            # 处理不同的数据加载器返回格式
            if len(batch) == 2:
                images, labels = batch
            elif len(batch) == 3:
                images, labels, _ = batch
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 计算准确率
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    
    # 生成分类报告
    report = classification_report(
        all_labels, all_predictions, 
        target_names=class_names, 
        output_dict=True
    )
    
    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    
    return {
        'accuracy': accuracy,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probs,
        'classification_report': report,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(cm, class_names, save_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('SE-Net 混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_detailed_results(results, class_names, output_dir):
    """保存详细结果"""
    # 保存分类报告
    report_path = output_dir / "classification_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results['classification_report'], f, indent=2, ensure_ascii=False)
    
    # 保存预测结果
    predictions_df = pd.DataFrame({
        'true_label': [class_names[label] for label in results['labels']],
        'predicted_label': [class_names[pred] for pred in results['predictions']],
        'true_label_idx': results['labels'],
        'predicted_label_idx': results['predictions'],
        'correct': np.array(results['labels']) == np.array(results['predictions'])
    })
    
    # 添加概率信息
    for i, class_name in enumerate(class_names):
        predictions_df[f'prob_{class_name}'] = [prob[i] for prob in results['probabilities']]
    
    predictions_path = output_dir / "detailed_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False, encoding='utf-8')
    
    # 保存错误分析
    errors_df = predictions_df[~predictions_df['correct']].copy()
    errors_path = output_dir / "error_analysis.csv"
    errors_df.to_csv(errors_path, index=False, encoding='utf-8')
    
    return len(errors_df)


def main():
    parser = argparse.ArgumentParser(description='Evaluate SE-Net model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to SE-Net checkpoint')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation/se_net_evaluation', 
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir + f"_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("🔥 SE-Net 模型评估开始")
    print("="*60)
    print(f"模型路径: {args.model_path}")
    print(f"输出目录: {output_dir}")
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据变换
    test_transform = transforms.Compose([
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载测试数据
    data_dir = Path("data/processed")
    test_df = pd.read_csv(data_dir / "test_split.csv")
    
    class_names = list(config.TARGET_CLASSES.values())
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    
    test_dataset = TomatoSpotDataset(test_df, test_transform, label_to_idx)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                           num_workers=config.NUM_WORKERS, pin_memory=True)
    
    print(f"测试数据集大小: {len(test_dataset)}")
    
    # 打印类别分布
    test_class_counts = test_df['label'].value_counts()
    print("测试集类别分布:")
    for class_name in class_names:
        count = test_class_counts.get(class_name, 0)
        print(f"  {class_name}: {count}")
    
    # 加载模型
    print("\n加载SE-Net模型...")
    model = load_se_net_model(args.model_path, device)
    
    # 评估模型
    print("\n开始模型评估...")
    results = evaluate_model(model, test_loader, device, class_names)
    
    # 保存结果
    print("\n保存评估结果...")
    
    # 绘制混淆矩阵
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(results['confusion_matrix'], class_names, cm_path)
    print(f"混淆矩阵已保存: {cm_path}")
    
    # 保存详细结果
    num_errors = save_detailed_results(results, class_names, output_dir)
    
    # 打印结果摘要
    print("\n" + "="*60)
    print("SE-Net 模型评估结果")
    print("="*60)
    print(f"模型: {args.model_path}")
    print(f"测试准确率: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"错误分类样本数: {num_errors}")
    
    # 打印每类指标
    report = results['classification_report']
    print(f"\n整体指标:")
    print(f"  宏平均精确率: {report['macro avg']['precision']:.4f}")
    print(f"  宏平均召回率: {report['macro avg']['recall']:.4f}")
    print(f"  宏平均F1分数: {report['macro avg']['f1-score']:.4f}")
    
    print(f"\n各类别指标:")
    for class_name in class_names:
        if class_name in report:
            metrics = report[class_name]
            print(f"  {class_name}:")
            print(f"    精确率: {metrics['precision']:.4f}")
            print(f"    召回率: {metrics['recall']:.4f}")
            print(f"    F1分数: {metrics['f1-score']:.4f}")
            print(f"    支持数: {int(metrics['support'])}")
    
    print(f"\n结果已保存到: {output_dir}")
    print("="*60)
    print("✅ SE-Net模型评估完成!")


if __name__ == "__main__":
    main() 