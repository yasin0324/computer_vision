#!/usr/bin/env python3
"""
注意力机制可视化分析脚本
对比SE-Net和CBAM的注意力机制，生成可视化分析报告
"""

import os
import sys
import torch
import argparse
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.attention_models import ResNetSE, ResNetCBAM
from src.data.dataset import TomatoSpotDataset
from src.config.config import Config
from src.visualization.attention_visualizer import AttentionVisualizer
from src.visualization.grad_cam import visualize_grad_cam

def load_model(model_type, checkpoint_path, config):
    """加载指定类型的模型"""
    print(f"加载{model_type}模型: {checkpoint_path}")
    
    if model_type == 'se_net':
        model = ResNetSE(
            num_classes=config.NUM_CLASSES,
            reduction=16,
            dropout_rate=0.7
        )
    elif model_type == 'cbam':
        model = ResNetCBAM(
            num_classes=config.NUM_CLASSES,
            reduction=16,
            dropout_rate=0.7
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"{model_type}模型加载成功")
    return model

def create_test_loader(config):
    """创建测试数据加载器"""
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
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                           num_workers=0, pin_memory=False)  # 单样本处理
    
    return test_loader, class_names

def visualize_single_sample(models, test_loader, class_names, output_dir, sample_idx=0):
    """可视化单个样本的注意力机制对比"""
    print(f"\n🔍 可视化样本 {sample_idx}")
    
    # 获取指定样本
    for i, batch in enumerate(test_loader):
        if i == sample_idx:
            if len(batch) == 2:
                image, label = batch
                image_path = f"sample_{sample_idx}"
            else:
                image, label, image_path = batch
            break
    else:
        print(f"样本 {sample_idx} 不存在")
        return
    
    sample_dir = Path(output_dir) / f"sample_{sample_idx}_comparison"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存原始图像信息
    class_name = class_names[label.item()]
    print(f"样本类别: {class_name}")
    
    # 为每个模型生成可视化
    for model_name, model in models.items():
        print(f"  处理 {model_name} 模型...")
        
        model_dir = sample_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        # 注意力可视化
        visualizer = AttentionVisualizer(model, device='cpu')
        try:
            visualizer.visualize_sample(image, label.item(), image_path, model_dir)
        except Exception as e:
            print(f"    注意力可视化失败: {e}")
        finally:
            visualizer.cleanup()
        
        # Grad-CAM可视化
        try:
            from src.visualization.grad_cam import GradCAM
            grad_cam = GradCAM(model, 'layer4', device='cpu')
            
            # 为真实类别生成CAM
            cam_path = model_dir / "grad_cam_true_class.png"
            grad_cam.save_visualization(image, label.item(), cam_path, class_names)
            
            # 为预测类别生成CAM
            with torch.no_grad():
                pred = model(image)
                pred_class = torch.argmax(pred, dim=1).item()
            
            if pred_class != label.item():
                cam_path = model_dir / "grad_cam_pred_class.png"
                grad_cam.save_visualization(image, pred_class, cam_path, class_names)
            
        except Exception as e:
            print(f"    Grad-CAM可视化失败: {e}")
    
    # 生成对比报告
    generate_comparison_report(models, image, label.item(), class_names, sample_dir)
    
    print(f"样本 {sample_idx} 可视化完成: {sample_dir}")

def generate_comparison_report(models, image, true_label, class_names, output_dir):
    """生成模型对比报告"""
    report_path = output_dir / "comparison_report.md"
    
    # 获取每个模型的预测
    predictions = {}
    for model_name, model in models.items():
        with torch.no_grad():
            pred = model(image)
            pred_probs = torch.softmax(pred, dim=1)[0]
            pred_class = torch.argmax(pred, dim=1).item()
            predictions[model_name] = {
                'class': pred_class,
                'confidence': pred_probs[pred_class].item(),
                'probabilities': pred_probs
            }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 注意力机制对比分析报告\n\n")
        
        f.write("## 样本信息\n")
        f.write(f"- 真实标签: {class_names[true_label]}\n")
        f.write(f"- 图像尺寸: {image.shape}\n\n")
        
        f.write("## 模型预测对比\n")
        f.write("| 模型 | 预测类别 | 置信度 | 预测正确 |\n")
        f.write("|------|----------|--------|----------|\n")
        
        for model_name, pred_info in predictions.items():
            pred_class_name = class_names[pred_info['class']]
            confidence = pred_info['confidence']
            is_correct = "✅" if pred_info['class'] == true_label else "❌"
            f.write(f"| {model_name} | {pred_class_name} | {confidence:.4f} | {is_correct} |\n")
        
        f.write("\n## 详细概率分布\n")
        for model_name, pred_info in predictions.items():
            f.write(f"\n### {model_name}\n")
            for i, class_name in enumerate(class_names):
                prob = pred_info['probabilities'][i].item()
                marker = "**" if i == true_label else ""
                f.write(f"- {marker}{class_name}{marker}: {prob:.4f} ({prob*100:.2f}%)\n")
        
        f.write("\n## 注意力机制分析\n")
        f.write("### SE-Net (通道注意力)\n")
        f.write("- 关注特征通道的重要性\n")
        f.write("- 学习'关注什么'特征\n")
        f.write("- 参数较少，计算高效\n\n")
        
        f.write("### CBAM (双重注意力)\n")
        f.write("- 结合通道注意力和空间注意力\n")
        f.write("- 学习'关注什么'和'关注哪里'\n")
        f.write("- 更全面的注意力机制\n\n")
        
        f.write("## 可视化文件说明\n")
        f.write("- `original_image.png`: 原始输入图像\n")
        f.write("- `channel_attention.png`: 通道注意力权重分布\n")
        f.write("- `spatial_attention.png`: 空间注意力热力图（仅CBAM）\n")
        f.write("- `feature_maps.png`: 特征图可视化\n")
        f.write("- `grad_cam_*.png`: Grad-CAM类激活图\n")
        f.write("- `attention_analysis.md`: 详细注意力分析\n")

def batch_visualize_attention(models, test_loader, class_names, output_dir, num_samples_per_class=2):
    """批量可视化不同类别的注意力机制"""
    print(f"\n📊 批量可视化注意力机制")
    
    # 为每个类别收集样本
    class_samples = {i: [] for i in range(len(class_names))}
    
    for batch in test_loader:
        if len(batch) == 2:
            image, label = batch
            image_path = f"sample_{len(class_samples[label.item()])}"
        else:
            image, label, image_path = batch
        
        label_idx = label.item()
        if len(class_samples[label_idx]) < num_samples_per_class:
            class_samples[label_idx].append((image, label_idx, image_path))
        
        # 检查是否收集够了样本
        if all(len(samples) >= num_samples_per_class for samples in class_samples.values()):
            break
    
    # 为每个类别和模型生成可视化
    for class_idx, class_name in enumerate(class_names):
        if not class_samples[class_idx]:
            continue
            
        print(f"\n处理类别: {class_name}")
        class_dir = Path(output_dir) / f"class_{class_name}"
        class_dir.mkdir(parents=True, exist_ok=True)
        
        for sample_idx, (image, label, image_path) in enumerate(class_samples[class_idx]):
            sample_dir = class_dir / f"sample_{sample_idx}"
            sample_dir.mkdir(exist_ok=True)
            
            for model_name, model in models.items():
                print(f"  {model_name} - 样本 {sample_idx}")
                
                model_dir = sample_dir / model_name
                model_dir.mkdir(exist_ok=True)
                
                # 注意力可视化
                visualizer = AttentionVisualizer(model, device='cpu')
                try:
                    visualizer.visualize_sample(image, label, image_path, model_dir)
                except Exception as e:
                    print(f"    注意力可视化失败: {e}")
                finally:
                    visualizer.cleanup()

def main():
    parser = argparse.ArgumentParser(description='注意力机制可视化分析')
    parser.add_argument('--se_net_path', type=str,
                       default='outputs/models/resnet50_se_net_from_baseline/best_checkpoint_epoch_20.pth',
                       help='SE-Net模型路径')
    parser.add_argument('--cbam_path', type=str,
                       default='outputs/models/resnet50_cbam_from_baseline/best_checkpoint_epoch_14.pth',
                       help='CBAM模型路径')
    parser.add_argument('--output_dir', type=str,
                       default='outputs/attention_visualization',
                       help='输出目录')
    parser.add_argument('--mode', type=str, choices=['single', 'batch', 'both'], default='both',
                       help='可视化模式')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='单样本模式下的样本索引')
    parser.add_argument('--num_samples', type=int, default=2,
                       help='批量模式下每个类别的样本数')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎨 注意力机制可视化分析")
    print("=" * 60)
    print(f"SE-Net模型: {args.se_net_path}")
    print(f"CBAM模型: {args.cbam_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"可视化模式: {args.mode}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载配置
    config = Config()
    
    # 加载模型
    models = {}
    
    if os.path.exists(args.se_net_path):
        models['SE-Net'] = load_model('se_net', args.se_net_path, config)
    else:
        print(f"⚠️ SE-Net模型文件不存在: {args.se_net_path}")
    
    if os.path.exists(args.cbam_path):
        models['CBAM'] = load_model('cbam', args.cbam_path, config)
    else:
        print(f"⚠️ CBAM模型文件不存在: {args.cbam_path}")
    
    if not models:
        print("❌ 没有可用的模型，退出")
        return
    
    # 创建测试数据加载器
    test_loader, class_names = create_test_loader(config)
    print(f"测试数据加载完成，类别: {class_names}")
    
    # 执行可视化
    if args.mode in ['single', 'both']:
        print(f"\n🔍 单样本可视化分析")
        visualize_single_sample(models, test_loader, class_names, 
                               args.output_dir, args.sample_idx)
    
    if args.mode in ['batch', 'both']:
        print(f"\n📊 批量可视化分析")
        batch_visualize_attention(models, test_loader, class_names,
                                 args.output_dir, args.num_samples)
    
    # 生成总结报告
    generate_summary_report(models, class_names, args.output_dir)
    
    print(f"\n🎉 注意力可视化分析完成！")
    print(f"结果保存在: {args.output_dir}")

def generate_summary_report(models, class_names, output_dir):
    """生成总结报告"""
    report_path = Path(output_dir) / "visualization_summary.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 注意力机制可视化分析总结\n\n")
        
        f.write("## 实验概述\n")
        f.write("本实验对比分析了SE-Net和CBAM两种注意力机制在植物叶片病害识别任务中的表现。\n\n")
        
        f.write("## 模型信息\n")
        for model_name in models.keys():
            f.write(f"- **{model_name}**: ResNet50 + {model_name}注意力机制\n")
        f.write(f"- **数据集**: 番茄叶斑病细粒度识别（{len(class_names)}类）\n")
        f.write(f"- **类别**: {', '.join(class_names)}\n\n")
        
        f.write("## 注意力机制对比\n")
        f.write("### SE-Net (Squeeze-and-Excitation)\n")
        f.write("- **机制**: 通道注意力\n")
        f.write("- **原理**: 通过全局平均池化和全连接层学习通道权重\n")
        f.write("- **优势**: 参数少，计算高效，关注重要特征通道\n")
        f.write("- **可视化**: 通道权重分布图\n\n")
        
        f.write("### CBAM (Convolutional Block Attention Module)\n")
        f.write("- **机制**: 通道注意力 + 空间注意力\n")
        f.write("- **原理**: 先学习通道权重，再学习空间位置权重\n")
        f.write("- **优势**: 更全面的注意力，同时关注'什么'和'哪里'\n")
        f.write("- **可视化**: 通道权重分布图 + 空间注意力热力图\n\n")
        
        f.write("## 可视化内容说明\n")
        f.write("### 文件结构\n")
        f.write("```\n")
        f.write("attention_visualization/\n")
        f.write("├── sample_X_comparison/          # 单样本对比分析\n")
        f.write("│   ├── SE-Net/                   # SE-Net可视化结果\n")
        f.write("│   ├── CBAM/                     # CBAM可视化结果\n")
        f.write("│   └── comparison_report.md      # 对比分析报告\n")
        f.write("├── class_XXX/                    # 各类别批量分析\n")
        f.write("│   └── sample_X/\n")
        f.write("│       ├── SE-Net/\n")
        f.write("│       └── CBAM/\n")
        f.write("└── visualization_summary.md      # 总结报告\n")
        f.write("```\n\n")
        
        f.write("### 可视化图像说明\n")
        f.write("- **original_image.png**: 原始输入图像\n")
        f.write("- **channel_attention.png**: 通道注意力权重柱状图\n")
        f.write("- **spatial_attention.png**: 空间注意力热力图（仅CBAM）\n")
        f.write("- **feature_maps.png**: 不同层的特征图可视化\n")
        f.write("- **grad_cam_*.png**: Grad-CAM类激活图\n")
        f.write("- **attention_analysis.md**: 详细数值分析\n\n")
        
        f.write("## 分析要点\n")
        f.write("1. **通道注意力对比**: 观察SE-Net和CBAM的通道权重分布差异\n")
        f.write("2. **空间注意力分析**: CBAM的空间注意力如何定位关键区域\n")
        f.write("3. **预测一致性**: 两种模型在相同样本上的预测差异\n")
        f.write("4. **类别特异性**: 不同病害类别激活的注意力模式\n")
        f.write("5. **错误案例分析**: 模型预测错误时的注意力模式\n\n")
        
        f.write("## 使用建议\n")
        f.write("1. 首先查看单样本对比分析，理解两种注意力机制的差异\n")
        f.write("2. 浏览各类别的批量分析，发现类别特异的注意力模式\n")
        f.write("3. 重点关注预测错误的样本，分析注意力机制的局限性\n")
        f.write("4. 结合Grad-CAM和注意力权重，全面理解模型决策过程\n")

if __name__ == "__main__":
    main() 