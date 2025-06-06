#!/usr/bin/env python3
"""
Grad-CAM可视化器
用于生成类激活图，显示模型关注的区域
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image

class GradCAM:
    """Grad-CAM可视化器"""
    
    def __init__(self, model, target_layer_name, device='cpu'):
        """
        初始化Grad-CAM
        
        Args:
            model: 目标模型
            target_layer_name: 目标层名称（如'layer4'）
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # 找到目标层
        self.target_layer = self._find_target_layer(target_layer_name)
        if self.target_layer is None:
            raise ValueError(f"Target layer '{target_layer_name}' not found in model")
        
        # 存储梯度和特征图
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        self._register_hooks()
    
    def _find_target_layer(self, layer_name):
        """查找目标层"""
        # 如果模型有model属性（如ResNetSE, ResNetCBAM）
        if hasattr(self.model, 'model'):
            model = self.model.model
        else:
            model = self.model
        
        # 查找层
        for name, module in model.named_modules():
            if name == layer_name:
                return module
        
        # 如果没找到，尝试直接访问属性
        if hasattr(model, layer_name):
            return getattr(model, layer_name)
        
        return None
    
    def _register_hooks(self):
        """注册前向和反向钩子"""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, class_idx=None):
        """
        生成类激活图
        
        Args:
            input_tensor: 输入张量 [1, 3, H, W]
            class_idx: 目标类别索引，如果为None则使用预测类别
        
        Returns:
            cam: 类激活图 [H, W]
            prediction: 预测结果
        """
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_()
        
        # 前向传播
        output = self.model(input_tensor)
        
        # 如果没有指定类别，使用预测类别
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        
        # 反向传播
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward(retain_graph=True)
        
        # 计算权重（梯度的全局平均池化）
        if self.gradients is not None:
            weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
            
            # 加权求和
            cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
            
            # ReLU激活
            cam = F.relu(cam)
            
            # 归一化到[0, 1]
            cam = cam - cam.min()
            cam = cam / cam.max() if cam.max() > 0 else cam
            
            return cam.detach().cpu().numpy(), output.detach().cpu()
        else:
            # 如果没有梯度，返回空的CAM
            print("Warning: No gradients captured, returning empty CAM")
            return np.zeros((7, 7)), output.detach().cpu()
    
    def visualize_cam(self, input_tensor, cam, original_size=(224, 224)):
        """
        可视化类激活图
        
        Args:
            input_tensor: 原始输入张量
            cam: 类激活图
            original_size: 原始图像尺寸
        
        Returns:
            heatmap: 热力图
            superimposed: 叠加图像
        """
        # 上采样CAM到原始图像尺寸
        cam_resized = cv2.resize(cam, original_size)
        
        # 转换为热力图
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # 获取原始图像
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        original_img = input_tensor.squeeze(0).cpu() * std + mean
        original_img = torch.clamp(original_img, 0, 1)
        original_img = original_img.permute(1, 2, 0).numpy()
        original_img = (original_img * 255).astype(np.uint8)
        
        # 叠加图像
        superimposed = heatmap * 0.4 + original_img * 0.6
        superimposed = superimposed.astype(np.uint8)
        
        return heatmap, superimposed, original_img
    
    def save_visualization(self, input_tensor, class_idx, output_path, class_names=None):
        """
        保存可视化结果
        
        Args:
            input_tensor: 输入张量
            class_idx: 目标类别索引
            output_path: 输出路径
            class_names: 类别名称列表
        """
        # 生成CAM
        cam, prediction = self.generate_cam(input_tensor, class_idx)
        
        # 可视化
        heatmap, superimposed, original_img = self.visualize_cam(input_tensor, cam)
        
        # 预测信息
        pred_probs = F.softmax(prediction, dim=1)[0]
        pred_class = torch.argmax(prediction, dim=1).item()
        
        # 创建图像
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始图像
        axes[0].imshow(original_img)
        axes[0].set_title('原始图像')
        axes[0].axis('off')
        
        # 热力图
        axes[1].imshow(heatmap)
        axes[1].set_title('Grad-CAM 热力图')
        axes[1].axis('off')
        
        # 叠加图像
        axes[2].imshow(superimposed)
        if class_names:
            title = f'叠加图像\n预测: {class_names[pred_class]} ({pred_probs[pred_class]:.3f})'
            if class_idx != pred_class and class_idx < len(class_names):
                title += f'\n目标: {class_names[class_idx]}'
        else:
            title = f'叠加图像\n预测: {pred_class} ({pred_probs[pred_class]:.3f})'
            if class_idx != pred_class:
                title += f'\n目标: {class_idx}'
        axes[2].set_title(title)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return cam, prediction


def visualize_grad_cam(model, test_loader, class_names, output_dir, 
                      target_layer='layer4', num_samples=3):
    """
    批量生成Grad-CAM可视化
    
    Args:
        model: 目标模型
        test_loader: 测试数据加载器
        class_names: 类别名称列表
        output_dir: 输出目录
        target_layer: 目标层名称
        num_samples: 每个类别的样本数
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建Grad-CAM可视化器
    grad_cam = GradCAM(model, target_layer)
    
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
    
    # 为每个类别生成Grad-CAM
    all_cams = {}
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = output_dir / f"class_{class_name}"
        class_dir.mkdir(exist_ok=True)
        
        print(f"正在生成 {class_name} 的Grad-CAM...")
        
        class_cams = []
        
        for sample_idx, (image, label, path) in enumerate(class_samples[class_idx]):
            # 为真实类别生成CAM
            output_path = class_dir / f"sample_{sample_idx}_true_class.png"
            cam, pred = grad_cam.save_visualization(
                image, label, output_path, class_names
            )
            class_cams.append(cam)
            
            # 为预测类别生成CAM（如果不同）
            pred_class = torch.argmax(pred, dim=1).item()
            if pred_class != label:
                output_path = class_dir / f"sample_{sample_idx}_pred_class.png"
                grad_cam.save_visualization(
                    image, pred_class, output_path, class_names
                )
        
        all_cams[class_name] = class_cams
    
    # 生成类别对比图
    _generate_class_comparison(all_cams, class_names, output_dir)
    
    print(f"Grad-CAM可视化完成！结果保存在: {output_dir}")
    return output_dir


def _generate_class_comparison(all_cams, class_names, output_dir):
    """生成类别对比图"""
    if not all_cams:
        return
    
    # 计算每个类别的平均CAM
    avg_cams = {}
    for class_name, cams in all_cams.items():
        if cams:
            avg_cam = np.mean(cams, axis=0)
            avg_cams[class_name] = avg_cam
    
    if len(avg_cams) < 2:
        return
    
    # 创建对比图
    num_classes = len(avg_cams)
    fig, axes = plt.subplots(1, num_classes, figsize=(5*num_classes, 5))
    
    if num_classes == 1:
        axes = [axes]
    
    for idx, (class_name, avg_cam) in enumerate(avg_cams.items()):
        im = axes[idx].imshow(avg_cam, cmap='jet')
        axes[idx].set_title(f'{class_name}\n平均激活图')
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_dir / "class_comparison.png", dpi=300, bbox_inches='tight')
    plt.close() 