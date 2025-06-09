#!/usr/bin/env python3
"""
统一的注意力机制可视化脚本
整合了简单可视化、详细对比分析和综合报告生成功能
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
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.attention_models import ResNetSE, ResNetCBAM
from src.data.dataset import TomatoSpotDataset
from src.config.config import Config

class UnifiedAttentionVisualizer:
    """统一的注意力可视化器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.class_names = list(config.TARGET_CLASSES.values())
        self.label_to_idx = {label: idx for idx, label in enumerate(self.class_names)}
        
        # 设置日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def load_models(self, model_configs):
        """批量加载模型"""
        for model_name, config in model_configs.items():
            try:
                model = self._load_single_model(config['type'], config['path'])
                self.models[model_name] = model
                self.logger.info(f"成功加载模型: {model_name}")
            except Exception as e:
                self.logger.error(f"加载模型失败 {model_name}: {e}")
    
    def _load_single_model(self, model_type, checkpoint_path):
        """加载单个模型"""
        if model_type == 'se_net':
            model = ResNetSE(
                num_classes=self.config.NUM_CLASSES,
                reduction=16,
                dropout_rate=0.7
            )
        elif model_type == 'cbam':
            model = ResNetCBAM(
                num_classes=self.config.NUM_CLASSES,
                reduction=16,
                dropout_rate=0.7
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def visualize_sample(self, sample_idx, output_dir, mode='comprehensive'):
        """可视化单个样本"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建数据加载器
        test_loader = self._create_data_loader('test')
        
        # 获取样本
        for i, batch in enumerate(test_loader):
            if i == sample_idx:
                image, label = batch[0], batch[1]
                break
        else:
            raise ValueError(f"样本 {sample_idx} 不存在")
        
        sample_dir = output_dir / f"sample_{sample_idx}_{mode}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"开始可视化样本 {sample_idx}, 模式: {mode}")
        
        # 保存原始图像
        self._save_original_image(image, sample_dir)
        
        # 执行可视化
        if mode == 'simple':
            self._simple_visualization(image, label, sample_dir)
        elif mode == 'detailed':
            self._detailed_visualization(image, label, sample_dir)
        else:  # comprehensive
            self._comprehensive_visualization(image, label, sample_dir)
        
        return sample_dir
    
    def _create_data_loader(self, data_type):
        """创建数据加载器"""
        transform = transforms.Compose([
            transforms.Resize((self.config.INPUT_SIZE, self.config.INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        data_dir = Path("data/processed")
        df = pd.read_csv(data_dir / f"{data_type}_split.csv")
        dataset = TomatoSpotDataset(df, transform, self.label_to_idx)
        
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    def _save_original_image(self, image_tensor, output_dir):
        """保存原始图像"""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        image = image_tensor.squeeze(0) * std + mean
        image = torch.clamp(image, 0, 1)
        
        image_np = image.permute(1, 2, 0).numpy()
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
        image_pil.save(output_dir / "original_image.png")
    
    def _simple_visualization(self, image, label, output_dir):
        """简单可视化模式"""
        results = {}
        
        for model_name, model in self.models.items():
            # 获取预测
            with torch.no_grad():
                pred = model(image)
                pred_probs = torch.softmax(pred, dim=1)
                pred_class = torch.argmax(pred, dim=1).item()
            
            results[model_name] = {
                'pred_class': pred_class,
                'confidence': pred_probs[0][pred_class].item()
            }
        
        # 生成报告
        self._generate_simple_report(results, label.item(), output_dir)
    
    def _detailed_visualization(self, image, label, output_dir):
        """详细可视化模式"""
        for model_name, model in self.models.items():
            model_dir = output_dir / model_name
            model_dir.mkdir(exist_ok=True)
            
            # 注意力权重提取和可视化
            attention_weights = self._extract_attention_weights(model, image)
            self._plot_attention_weights(attention_weights, model_dir, model_name)
        
        # 生成对比报告
        self._generate_comparison_report(image, label.item(), output_dir)
    
    def _comprehensive_visualization(self, image, label, output_dir):
        """综合可视化模式"""
        self._detailed_visualization(image, label, output_dir)
        self._generate_comprehensive_analysis(label.item(), output_dir)
    
    def _extract_attention_weights(self, model, input_tensor):
        """提取注意力权重"""
        attention_weights = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if any(keyword in name.lower() for keyword in ['se', 'cbam', 'attention']):
                    attention_weights[name] = output.detach().cpu()
            return hook
        
        hooks = []
        base_model = model.model if hasattr(model, 'model') else model
        
        for name, module in base_model.named_modules():
            if any(keyword in name.lower() for keyword in ['se', 'cbam', 'attention']):
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        with torch.no_grad():
            _ = model(input_tensor)
        
        for hook in hooks:
            hook.remove()
        
        return attention_weights
    
    def _plot_attention_weights(self, attention_weights, output_dir, model_name):
        """绘制注意力权重"""
        if not attention_weights:
            return
        
        num_weights = min(len(attention_weights), 4)
        fig, axes = plt.subplots(1, num_weights, figsize=(15, 4))
        if num_weights == 1:
            axes = [axes]
        
        for idx, (name, weights) in enumerate(list(attention_weights.items())[:4]):
            ax = axes[idx]
            
            if weights.dim() == 4 and weights.shape[2] == 1 and weights.shape[3] == 1:
                # 通道注意力
                weights_1d = weights.squeeze().numpy()
                ax.bar(range(len(weights_1d)), weights_1d)
                ax.set_title(f'{name}\nChannel Attention')
            else:
                # 空间注意力
                weights_2d = weights.squeeze().numpy()
                if weights_2d.ndim > 2:
                    weights_2d = weights_2d.mean(axis=0)
                
                im = ax.imshow(weights_2d, cmap='jet')
                ax.set_title(f'{name}\nSpatial Attention')
                plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{model_name}_attention.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_simple_report(self, results, true_label, output_dir):
        """生成简单报告"""
        with open(output_dir / "simple_report.md", 'w', encoding='utf-8') as f:
            f.write("# 简单注意力分析报告\n\n")
            f.write(f"真实标签: {self.class_names[true_label]}\n\n")
            
            for model_name, result in results.items():
                pred_class = result['pred_class']
                confidence = result['confidence']
                correct = "✅" if pred_class == true_label else "❌"
                
                f.write(f"## {model_name}\n")
                f.write(f"- 预测: {self.class_names[pred_class]}\n")
                f.write(f"- 置信度: {confidence:.4f}\n")
                f.write(f"- 正确性: {correct}\n\n")
    
    def _generate_comparison_report(self, image, true_label, output_dir):
        """生成对比报告"""
        predictions = {}
        for model_name, model in self.models.items():
            with torch.no_grad():
                pred = model(image)
                pred_probs = torch.softmax(pred, dim=1)[0]
                pred_class = torch.argmax(pred, dim=1).item()
                predictions[model_name] = {
                    'class': pred_class,
                    'confidence': pred_probs[pred_class].item(),
                    'probabilities': pred_probs
                }
        
        with open(output_dir / "comparison_report.md", 'w', encoding='utf-8') as f:
            f.write("# 模型对比分析报告\n\n")
            f.write(f"真实标签: {self.class_names[true_label]}\n\n")
            
            f.write("| 模型 | 预测类别 | 置信度 | 正确性 |\n")
            f.write("|------|----------|--------|--------|\n")
            
            for model_name, pred_info in predictions.items():
                pred_class_name = self.class_names[pred_info['class']]
                confidence = pred_info['confidence']
                correct = "✅" if pred_info['class'] == true_label else "❌"
                f.write(f"| {model_name} | {pred_class_name} | {confidence:.4f} | {correct} |\n")
    
    def _generate_comprehensive_analysis(self, true_label, output_dir):
        """生成综合分析"""
        with open(output_dir / "comprehensive_analysis.md", 'w', encoding='utf-8') as f:
            f.write("# 综合注意力分析报告\n\n")
            f.write("## 注意力机制对比\n\n")
            f.write("### SE-Net\n")
            f.write("- 机制: 通道注意力\n")
            f.write("- 特点: 学习'关注什么特征'\n\n")
            f.write("### CBAM\n")
            f.write("- 机制: 通道 + 空间注意力\n")
            f.write("- 特点: 学习'关注什么特征'和'关注哪个位置'\n\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='统一注意力可视化工具')
    parser.add_argument('--models', type=str, required=True, help='模型配置JSON文件')
    parser.add_argument('--sample_idx', type=int, default=0, help='样本索引')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--mode', type=str, default='comprehensive',
                       choices=['simple', 'detailed', 'comprehensive'], help='可视化模式')
    
    args = parser.parse_args()
    
    config = Config()
    visualizer = UnifiedAttentionVisualizer(config)
    
    # 加载模型配置
    import json
    with open(args.models, 'r') as f:
        model_configs = json.load(f)
    
    visualizer.load_models(model_configs)
    visualizer.visualize_sample(args.sample_idx, args.output_dir, args.mode)


if __name__ == "__main__":
    main() 