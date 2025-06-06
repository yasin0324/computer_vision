#!/usr/bin/env python3
"""
简化的注意力可视化脚本
专注于基本的注意力权重可视化
"""

import os
import sys
import torch
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.attention_models import ResNetSE, ResNetCBAM
from src.data.dataset import TomatoSpotDataset
from src.config.config import Config

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

def extract_attention_weights(model, input_tensor):
    """提取注意力权重"""
    attention_weights = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            # 对于SE模块，直接保存输出作为注意力权重
            if 'se' in name.lower():
                attention_weights[name] = output.detach().cpu()
            # 对于CBAM模块，需要特殊处理
            elif hasattr(module, 'channel_attention') or hasattr(module, 'spatial_attention'):
                attention_weights[name] = output.detach().cpu()
        return hook
    
    # 注册钩子
    hooks = []
    if hasattr(model, 'model'):
        base_model = model.model
    else:
        base_model = model
    
    # 为关键层注册钩子
    for name, module in base_model.named_modules():
        if 'se' in name or 'cbam' in name or 'attention' in name:
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)
    
    # 前向传播
    with torch.no_grad():
        output = model(input_tensor)
        pred_probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(output, dim=1).item()
    
    # 清理钩子
    for hook in hooks:
        hook.remove()
    
    return attention_weights, pred_probs, pred_class

def visualize_sample_simple(models, sample_data, class_names, output_dir, sample_idx):
    """简化的样本可视化"""
    image, label, image_path = sample_data
    
    # 确保image_path是字符串
    if not isinstance(image_path, str):
        image_path = f"sample_{sample_idx}"
    
    sample_dir = Path(output_dir) / f"sample_{sample_idx}_simple"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存原始图像
    save_original_image(image, sample_dir)
    
    # 为每个模型提取和可视化注意力
    results = {}
    
    for model_name, model in models.items():
        print(f"  处理 {model_name} 模型...")
        
        # 提取注意力权重
        attention_weights, pred_probs, pred_class = extract_attention_weights(model, image)
        
        results[model_name] = {
            'attention_weights': attention_weights,
            'pred_probs': pred_probs,
            'pred_class': pred_class
        }
        
        # 可视化注意力权重
        model_dir = sample_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        visualize_attention_weights(attention_weights, model_dir, model_name)
    
    # 生成对比报告
    generate_simple_report(results, label.item(), class_names, sample_dir, image_path)
    
    print(f"样本 {sample_idx} 简化可视化完成: {sample_dir}")
    return sample_dir

def save_original_image(image_tensor, output_dir):
    """保存原始图像"""
    # 反标准化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    image = image_tensor.squeeze(0) * std + mean
    image = torch.clamp(image, 0, 1)
    
    # 转换为PIL图像并保存
    image_np = image.permute(1, 2, 0).numpy()
    image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
    image_pil.save(output_dir / "original_image.png")

def visualize_attention_weights(attention_weights, output_dir, model_name):
    """可视化注意力权重"""
    if not attention_weights:
        print(f"    {model_name}: 没有找到注意力权重")
        return
    
    # 创建图像
    num_weights = len(attention_weights)
    if num_weights == 0:
        return
    
    fig, axes = plt.subplots(1, min(num_weights, 4), figsize=(15, 4))
    if num_weights == 1:
        axes = [axes]
    elif num_weights > 4:
        axes = axes[:4]
        attention_weights = dict(list(attention_weights.items())[:4])
    
    for idx, (name, weights) in enumerate(attention_weights.items()):
        ax = axes[idx] if num_weights > 1 else axes[0]
        
        # 处理不同形状的权重
        if weights.dim() == 4:  # [B, C, H, W]
            if weights.shape[2] == 1 and weights.shape[3] == 1:  # 通道注意力
                weights_1d = weights.squeeze().numpy()
                bars = ax.bar(range(len(weights_1d)), weights_1d)
                ax.set_title(f'{name}\nChannel Attention')
                ax.set_xlabel('Channel Index')
                ax.set_ylabel('Weight')
                
                # 高亮最重要的通道
                if len(weights_1d) > 5:
                    max_indices = np.argsort(weights_1d)[-5:]
                    for i in max_indices:
                        bars[i].set_color('red')
            else:  # 空间注意力或特征图
                # 如果是3D，取第一个通道或平均
                if weights.dim() == 4 and weights.shape[1] > 1:
                    weights_2d = weights.squeeze(0).mean(dim=0).numpy()
                else:
                    weights_2d = weights.squeeze().numpy()
                    
                # 确保是2D
                if weights_2d.ndim > 2:
                    weights_2d = weights_2d.mean(axis=0)
                
                im = ax.imshow(weights_2d, cmap='jet')
                ax.set_title(f'{name}\nSpatial Attention')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.axis('off')
        else:
            # 其他情况，尝试展平并显示
            weights_flat = weights.flatten().numpy()
            ax.plot(weights_flat)
            ax.set_title(f'{name}\nWeight Distribution')
    
    # 隐藏多余的子图
    for idx in range(len(attention_weights), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_attention_weights.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    {model_name}: 保存了 {len(attention_weights)} 个注意力权重图")

def generate_simple_report(results, true_label, class_names, output_dir, image_path):
    """生成简化报告"""
    report_path = output_dir / "simple_analysis.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 简化注意力分析报告\n\n")
        
        f.write("## 样本信息\n")
        f.write(f"- 图像路径: {image_path}\n")
        f.write(f"- 真实标签: {class_names[true_label]}\n\n")
        
        f.write("## 模型预测对比\n")
        f.write("| 模型 | 预测类别 | 置信度 | 预测正确 |\n")
        f.write("|------|----------|--------|----------|\n")
        
        for model_name, result in results.items():
            pred_class = result['pred_class']
            confidence = result['pred_probs'][0][pred_class].item()
            pred_class_name = class_names[pred_class]
            is_correct = "✅" if pred_class == true_label else "❌"
            f.write(f"| {model_name} | {pred_class_name} | {confidence:.4f} | {is_correct} |\n")
        
        f.write("\n## 详细概率分布\n")
        for model_name, result in results.items():
            f.write(f"\n### {model_name}\n")
            pred_probs = result['pred_probs'][0]
            for i, class_name in enumerate(class_names):
                prob = pred_probs[i].item()
                marker = "**" if i == true_label else ""
                f.write(f"- {marker}{class_name}{marker}: {prob:.4f} ({prob*100:.2f}%)\n")
        
        f.write("\n## 注意力权重统计\n")
        for model_name, result in results.items():
            f.write(f"\n### {model_name}\n")
            attention_weights = result['attention_weights']
            if attention_weights:
                f.write(f"- 检测到 {len(attention_weights)} 个注意力模块\n")
                for name, weights in attention_weights.items():
                    f.write(f"  - {name}: 形状 {list(weights.shape)}\n")
            else:
                f.write("- 未检测到注意力权重\n")

def main():
    parser = argparse.ArgumentParser(description='简化注意力可视化')
    parser.add_argument('--se_net_path', type=str,
                       default='outputs/models/resnet50_se_net_from_baseline/best_checkpoint_epoch_20.pth',
                       help='SE-Net模型路径')
    parser.add_argument('--cbam_path', type=str,
                       default='outputs/models/resnet50_cbam_from_baseline/best_checkpoint_epoch_14.pth',
                       help='CBAM模型路径')
    parser.add_argument('--output_dir', type=str,
                       default='outputs/simple_attention_viz',
                       help='输出目录')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='样本索引')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎨 简化注意力可视化分析")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载配置
    config = Config()
    
    # 加载模型
    models = {}
    
    if os.path.exists(args.se_net_path):
        models['SE-Net'] = load_model('se_net', args.se_net_path, config)
    
    if os.path.exists(args.cbam_path):
        models['CBAM'] = load_model('cbam', args.cbam_path, config)
    
    if not models:
        print("❌ 没有可用的模型")
        return
    
    # 创建测试数据加载器
    test_transform = transforms.Compose([
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    data_dir = Path("data/processed")
    test_df = pd.read_csv(data_dir / "test_split.csv")
    
    class_names = list(config.TARGET_CLASSES.values())
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    
    test_dataset = TomatoSpotDataset(test_df, test_transform, label_to_idx)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # 获取指定样本
    for i, batch in enumerate(test_loader):
        if i == args.sample_idx:
            if len(batch) == 2:
                image, label = batch
                image_path = f"sample_{args.sample_idx}"
            else:
                image, label, image_path = batch
            break
    else:
        print(f"样本 {args.sample_idx} 不存在")
        return
    
    print(f"处理样本 {args.sample_idx}: {class_names[label.item()]}")
    
    # 可视化样本
    visualize_sample_simple(models, (image, label, image_path), class_names, 
                           args.output_dir, args.sample_idx)
    
    print(f"\n🎉 简化可视化完成！结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main() 