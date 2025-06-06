#!/usr/bin/env python3
"""
注意力机制可视化器
用于可视化SE-Net和CBAM的注意力权重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
from PIL import Image

class AttentionVisualizer:
    """注意力机制可视化器"""
    
    def __init__(self, model, device='cpu'):
        """
        初始化可视化器
        
        Args:
            model: 带注意力机制的模型
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # 存储注意力权重的钩子
        self.attention_weights = {}
        self.feature_maps = {}
        self.hooks = []
        
        # 注册钩子函数
        self._register_hooks()
    
    def _register_hooks(self):
        """注册钩子函数来捕获注意力权重"""
        
        def save_attention_hook(name):
            def hook(module, input, output):
                if hasattr(module, 'channel_attention'):
                    # CBAM的通道注意力
                    if hasattr(module.channel_attention, 'last_attention'):
                        self.attention_weights[f'{name}_channel'] = module.channel_attention.last_attention
                elif hasattr(module, 'se'):
                    # SE注意力
                    if hasattr(module.se, 'last_attention'):
                        self.attention_weights[f'{name}_se'] = module.se.last_attention
                
                if hasattr(module, 'spatial_attention'):
                    # CBAM的空间注意力
                    if hasattr(module.spatial_attention, 'last_attention'):
                        self.attention_weights[f'{name}_spatial'] = module.spatial_attention.last_attention
                
                # 保存特征图
                self.feature_maps[name] = output.detach()
            return hook
        
        # 为ResNet的各层注册钩子
        if hasattr(self.model, 'model'):
            model = self.model.model
        else:
            model = self.model
            
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            if hasattr(model, layer_name):
                layer = getattr(model, layer_name)
                for i, block in enumerate(layer):
                    hook_name = f'{layer_name}_{i}'
                    hook = block.register_forward_hook(save_attention_hook(hook_name))
                    self.hooks.append(hook)
    
    def _modify_attention_modules(self):
        """修改注意力模块以保存权重"""
        
        def modify_se_module(se_module):
            original_forward = se_module.forward
            def new_forward(x):
                # 执行原始forward
                out = original_forward(x)
                # 保存注意力权重
                se_module.last_attention = out.detach()
                return out
            se_module.forward = new_forward
        
        def modify_channel_attention(ca_module):
            original_forward = ca_module.forward
            def new_forward(x):
                out = original_forward(x)
                ca_module.last_attention = out.detach()
                return out
            ca_module.forward = new_forward
        
        def modify_spatial_attention(sa_module):
            original_forward = sa_module.forward
            def new_forward(x):
                out = original_forward(x)
                sa_module.last_attention = out.detach()
                return out
            sa_module.forward = new_forward
        
        # 遍历模型修改注意力模块
        if hasattr(self.model, 'model'):
            model = self.model.model
        else:
            model = self.model
            
        for name, module in model.named_modules():
            if hasattr(module, 'se'):
                modify_se_module(module.se)
            if hasattr(module, 'channel_attention'):
                modify_channel_attention(module.channel_attention)
            if hasattr(module, 'spatial_attention'):
                modify_spatial_attention(module.spatial_attention)
    
    def visualize_sample(self, image_tensor, label, image_path, output_dir):
        """
        可视化单个样本的注意力
        
        Args:
            image_tensor: 输入图像张量 [1, 3, H, W]
            label: 真实标签
            image_path: 图像路径
            output_dir: 输出目录
        """
        self.attention_weights.clear()
        self.feature_maps.clear()
        
        # 确保image_path是字符串
        if not isinstance(image_path, str):
            image_path = str(image_path)
        
        # 修改注意力模块
        self._modify_attention_modules()
        
        # 前向传播
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            logits = self.model(image_tensor)
            pred_prob = F.softmax(logits, dim=1)
            pred_class = torch.argmax(logits, dim=1).item()
        
        # 创建输出目录
        sample_dir = Path(output_dir) / f"sample_{Path(image_path).stem}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存原始图像
        self._save_original_image(image_tensor, sample_dir)
        
        # 可视化通道注意力
        self._visualize_channel_attention(sample_dir)
        
        # 可视化空间注意力
        self._visualize_spatial_attention(image_tensor, sample_dir)
        
        # 可视化特征图
        self._visualize_feature_maps(sample_dir)
        
        # 生成综合报告
        self._generate_attention_report(
            image_path, label, pred_class, pred_prob, sample_dir
        )
        
        return sample_dir
    
    def _save_original_image(self, image_tensor, output_dir):
        """保存原始图像"""
        # 反标准化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        image = image_tensor.squeeze(0).cpu() * std + mean
        image = torch.clamp(image, 0, 1)
        
        # 转换为PIL图像并保存
        image_np = image.permute(1, 2, 0).numpy()
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
        image_pil.save(output_dir / "original_image.png")
    
    def _visualize_channel_attention(self, output_dir):
        """可视化通道注意力权重"""
        channel_attentions = {k: v for k, v in self.attention_weights.items() 
                            if 'channel' in k or 'se' in k}
        
        if not channel_attentions:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, (name, attention) in enumerate(channel_attentions.items()):
            if idx >= 4:
                break
                
            # 获取通道注意力权重 [1, C, 1, 1] -> [C]
            weights = attention.squeeze().cpu().numpy()
            
            ax = axes[idx]
            bars = ax.bar(range(len(weights)), weights)
            ax.set_title(f'{name} 通道注意力权重')
            ax.set_xlabel('通道索引')
            ax.set_ylabel('注意力权重')
            
            # 高亮最重要的通道
            max_indices = np.argsort(weights)[-5:]
            for i in max_indices:
                bars[i].set_color('red')
        
        # 隐藏多余的子图
        for idx in range(len(channel_attentions), 4):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / "channel_attention.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_spatial_attention(self, image_tensor, output_dir):
        """可视化空间注意力权重"""
        spatial_attentions = {k: v for k, v in self.attention_weights.items() 
                            if 'spatial' in k}
        
        if not spatial_attentions:
            return
        
        # 获取原始图像
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        original_img = (image_tensor.squeeze(0).cpu() * std + mean).permute(1, 2, 0).numpy()
        original_img = np.clip(original_img, 0, 1)
        
        num_attentions = len(spatial_attentions)
        if num_attentions == 0:
            return
            
        fig, axes = plt.subplots(2, num_attentions, figsize=(5*num_attentions, 10))
        if num_attentions == 1:
            axes = axes.reshape(2, 1)
        
        for idx, (name, attention) in enumerate(spatial_attentions.items()):
            # 空间注意力 [1, 1, H, W] -> [H, W]
            spatial_map = attention.squeeze().cpu().numpy()
            
            # 上采样到原始图像尺寸
            spatial_map_resized = cv2.resize(spatial_map, (224, 224))
            
            # 显示原始图像
            axes[0, idx].imshow(original_img)
            axes[0, idx].set_title(f'原始图像')
            axes[0, idx].axis('off')
            
            # 显示空间注意力热力图
            im = axes[1, idx].imshow(spatial_map_resized, cmap='jet', alpha=0.7)
            axes[1, idx].imshow(original_img, alpha=0.3)
            axes[1, idx].set_title(f'{name} 空间注意力')
            axes[1, idx].axis('off')
            
            # 添加颜色条
            plt.colorbar(im, ax=axes[1, idx], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(output_dir / "spatial_attention.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_feature_maps(self, output_dir):
        """可视化特征图"""
        if not self.feature_maps:
            return
        
        # 选择几个关键层的特征图
        key_layers = ['layer1_2', 'layer2_3', 'layer3_5', 'layer4_2']
        available_layers = [layer for layer in key_layers if layer in self.feature_maps]
        
        if not available_layers:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, layer_name in enumerate(available_layers[:4]):
            feature_map = self.feature_maps[layer_name]
            
            # 取前16个通道的平均值
            feature_avg = feature_map.squeeze(0)[:16].mean(dim=0).cpu().numpy()
            
            im = axes[idx].imshow(feature_avg, cmap='viridis')
            axes[idx].set_title(f'{layer_name} 特征图')
            axes[idx].axis('off')
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        
        # 隐藏多余的子图
        for idx in range(len(available_layers), 4):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / "feature_maps.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_attention_report(self, image_path, true_label, pred_label, pred_prob, output_dir):
        """生成注意力分析报告"""
        class_names = ['bacterial_spot', 'septoria_leaf_spot', 'target_spot', 'healthy']
        
        report_path = output_dir / "attention_analysis.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 注意力机制分析报告\n\n")
            f.write(f"## 样本信息\n")
            f.write(f"- 图像路径: {image_path}\n")
            f.write(f"- 真实标签: {class_names[true_label]}\n")
            f.write(f"- 预测标签: {class_names[pred_label]}\n")
            f.write(f"- 预测正确: {'✅' if true_label == pred_label else '❌'}\n\n")
            
            f.write("## 预测概率分布\n")
            for i, class_name in enumerate(class_names):
                prob = pred_prob[0][i].item()
                f.write(f"- {class_name}: {prob:.4f} ({prob*100:.2f}%)\n")
            
            f.write("\n## 注意力权重统计\n")
            
            # 通道注意力统计
            channel_attentions = {k: v for k, v in self.attention_weights.items() 
                                if 'channel' in k or 'se' in k}
            if channel_attentions:
                f.write("### 通道注意力\n")
                for name, attention in channel_attentions.items():
                    weights = attention.squeeze().cpu().numpy()
                    f.write(f"- {name}:\n")
                    f.write(f"  - 最大权重: {weights.max():.4f}\n")
                    f.write(f"  - 最小权重: {weights.min():.4f}\n")
                    f.write(f"  - 平均权重: {weights.mean():.4f}\n")
                    f.write(f"  - 权重标准差: {weights.std():.4f}\n")
            
            # 空间注意力统计
            spatial_attentions = {k: v for k, v in self.attention_weights.items() 
                                if 'spatial' in k}
            if spatial_attentions:
                f.write("\n### 空间注意力\n")
                for name, attention in spatial_attentions.items():
                    spatial_map = attention.squeeze().cpu().numpy()
                    f.write(f"- {name}:\n")
                    f.write(f"  - 最大权重: {spatial_map.max():.4f}\n")
                    f.write(f"  - 最小权重: {spatial_map.min():.4f}\n")
                    f.write(f"  - 平均权重: {spatial_map.mean():.4f}\n")
                    f.write(f"  - 权重标准差: {spatial_map.std():.4f}\n")
    
    def cleanup(self):
        """清理钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def visualize_attention_maps(model, test_loader, class_names, output_dir, num_samples=5):
    """
    批量可视化注意力图
    
    Args:
        model: 带注意力机制的模型
        test_loader: 测试数据加载器
        class_names: 类别名称列表
        output_dir: 输出目录
        num_samples: 每个类别可视化的样本数
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = AttentionVisualizer(model)
    
    # 为每个类别收集样本
    class_samples = {i: [] for i in range(len(class_names))}
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 2:
                images, labels = batch
                paths = [f"sample_{i}" for i in range(len(images))]
            else:
                images, labels, paths = batch
            
            for img, label, path in zip(images, labels, paths):
                label_idx = label.item()
                if len(class_samples[label_idx]) < num_samples:
                    class_samples[label_idx].append((img.unsqueeze(0), label_idx, path))
            
            # 检查是否收集够了样本
            if all(len(samples) >= num_samples for samples in class_samples.values()):
                break
    
    # 可视化每个类别的样本
    for class_idx, class_name in enumerate(class_names):
        class_dir = output_dir / f"class_{class_name}"
        class_dir.mkdir(exist_ok=True)
        
        print(f"正在可视化类别: {class_name}")
        
        for sample_idx, (image, label, path) in enumerate(class_samples[class_idx]):
            print(f"  处理样本 {sample_idx + 1}/{num_samples}")
            sample_output_dir = visualizer.visualize_sample(
                image, label, path, class_dir / f"sample_{sample_idx}"
            )
    
    # 清理
    visualizer.cleanup()
    
    print(f"注意力可视化完成！结果保存在: {output_dir}")
    return output_dir 