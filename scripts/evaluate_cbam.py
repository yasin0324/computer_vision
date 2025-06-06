#!/usr/bin/env python3
"""
CBAM模型评估脚本
评估训练好的CBAM模型在测试集上的性能
"""

import os
import sys
import torch
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.attention_models import ResNetCBAM
from src.data.dataset import TomatoSpotDataset
from src.config.config import Config

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
    plt.title('CBAM 混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='评估CBAM模型')
    parser.add_argument('--model_path', type=str, 
                       default='outputs/models/resnet50_cbam_from_baseline/best_checkpoint_epoch_14.pth',
                       help='CBAM模型检查点路径')
    parser.add_argument('--output_dir', type=str, 
                       default='outputs/cbam_evaluation',
                       help='评估结果输出目录')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔍 CBAM模型测试集评估")
    print("=" * 60)
    print(f"模型路径: {args.model_path}")
    print(f"输出目录: {args.output_dir}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载配置
    config = Config()
    
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
    print("\n📊 加载测试数据...")
    data_dir = Path("data/processed")
    test_df = pd.read_csv(data_dir / "test_split.csv")
    
    class_names = list(config.TARGET_CLASSES.values())
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    
    test_dataset = TomatoSpotDataset(test_df, test_transform, label_to_idx)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                           num_workers=config.NUM_WORKERS, pin_memory=True)
    
    print(f"测试数据集大小: {len(test_dataset)}")
    
    # 创建CBAM模型
    print("🏗️ 创建CBAM模型...")
    model = ResNetCBAM(
        num_classes=config.NUM_CLASSES,
        reduction=16,
        dropout_rate=0.7
    )
    
    # 加载训练好的权重
    print(f"📥 加载模型权重: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # 评估模型
    print("\n🚀 开始测试集评估...")
    results = evaluate_model(model, test_loader, device, class_names)
    
    # 绘制混淆矩阵
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(results['confusion_matrix'], class_names, cm_path)
    print(f"混淆矩阵已保存: {cm_path}")
    
    # 显示结果摘要
    print("\n" + "=" * 60)
    print("📊 CBAM模型评估结果摘要")
    print("=" * 60)
    print(f"测试准确率: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"测试样本总数: {len(results['labels'])}")
    
    # 计算错误分类样本数
    misclassified_count = np.sum(np.array(results['labels']) != np.array(results['predictions']))
    print(f"错误分类样本: {misclassified_count}")
    print(f"错误率: {(1-results['accuracy'])*100:.2f}%")
    
    print(f"\n各类别性能:")
    for class_name in class_names:
        metrics = results['classification_report'][class_name]
        print(f"  {class_name}:")
        print(f"    精确率: {metrics['precision']:.4f}")
        print(f"    召回率: {metrics['recall']:.4f}")
        print(f"    F1分数: {metrics['f1-score']:.4f}")
    
    # 保存最终结果报告
    report_path = os.path.join(args.output_dir, 'cbam_final_results.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# CBAM模型最终评估结果\n\n")
        f.write("## 模型信息\n")
        f.write(f"- 模型类型: ResNet50 + CBAM\n")
        f.write(f"- 检查点: {args.model_path}\n")
        f.write(f"- 注意力机制: 通道注意力 + 空间注意力\n")
        f.write(f"- 降维比例: 16\n\n")
        
        f.write("## 测试集性能\n")
        f.write(f"- **测试准确率**: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n")
        f.write(f"- **测试样本总数**: {len(results['labels'])}\n")
        f.write(f"- **错误分类样本**: {misclassified_count}\n")
        f.write(f"- **错误率**: {(1-results['accuracy'])*100:.2f}%\n\n")
        
        f.write("## 各类别详细性能\n")
        f.write("| 类别 | 精确率 | 召回率 | F1分数 |\n")
        f.write("|------|--------|--------|--------|\n")
        for class_name in class_names:
            metrics = results['classification_report'][class_name]
            f.write(f"| {class_name} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1-score']:.4f} |\n")
    
    print(f"\n📄 详细结果报告已保存到: {report_path}")
    print("🎉 CBAM模型评估完成!")

if __name__ == "__main__":
    main() 