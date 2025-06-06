"""
注意力机制模块
实现SE-Net和CBAM注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    
    论文: Squeeze-and-Excitation Networks (CVPR 2018)
    """
    
    def __init__(self, channels, reduction=16):
        """
        Args:
            channels: 输入通道数
            reduction: 降维比例
        """
        super(SEBlock, self).__init__()
        self.channels = channels
        self.reduction = reduction
        
        # Squeeze: 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Excitation: 两个全连接层
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: 输入特征图 [B, C, H, W]
        Returns:
            输出特征图 [B, C, H, W]
        """
        batch_size, channels, _, _ = x.size()
        
        # Squeeze: [B, C, H, W] -> [B, C, 1, 1] -> [B, C]
        squeeze = self.global_avg_pool(x).view(batch_size, channels)
        
        # Excitation: [B, C] -> [B, C//r] -> [B, C]
        excitation = self.fc1(squeeze)
        excitation = self.relu(excitation)
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation)
        
        # Scale: [B, C] -> [B, C, 1, 1]
        excitation = excitation.view(batch_size, channels, 1, 1)
        
        # 应用注意力权重
        return x * excitation


class ChannelAttention(nn.Module):
    """
    CBAM的通道注意力模块
    """
    
    def __init__(self, channels, reduction=16):
        """
        Args:
            channels: 输入通道数
            reduction: 降维比例
        """
        super(ChannelAttention, self).__init__()
        self.channels = channels
        self.reduction = reduction
        
        # 全局平均池化和最大池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享的MLP
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: 输入特征图 [B, C, H, W]
        Returns:
            通道注意力权重 [B, C, 1, 1]
        """
        batch_size, channels, _, _ = x.size()
        
        # 平均池化分支
        avg_pool = self.global_avg_pool(x).view(batch_size, channels)
        avg_out = self.mlp(avg_pool)
        
        # 最大池化分支
        max_pool = self.global_max_pool(x).view(batch_size, channels)
        max_out = self.mlp(max_pool)
        
        # 合并并激活
        channel_attention = self.sigmoid(avg_out + max_out)
        return channel_attention.view(batch_size, channels, 1, 1)


class SpatialAttention(nn.Module):
    """
    CBAM的空间注意力模块
    """
    
    def __init__(self, kernel_size=7):
        """
        Args:
            kernel_size: 卷积核大小
        """
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size
        
        # 空间注意力卷积
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                             padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: 输入特征图 [B, C, H, W]
        Returns:
            空间注意力权重 [B, 1, H, W]
        """
        # 通道维度的平均和最大
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # 拼接
        spatial_input = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        
        # 卷积和激活
        spatial_attention = self.conv(spatial_input)
        spatial_attention = self.sigmoid(spatial_attention)
        
        return spatial_attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    
    论文: CBAM: Convolutional Block Attention Module (ECCV 2018)
    """
    
    def __init__(self, channels, reduction=16, kernel_size=7):
        """
        Args:
            channels: 输入通道数
            reduction: 通道注意力的降维比例
            kernel_size: 空间注意力的卷积核大小
        """
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        """
        Args:
            x: 输入特征图 [B, C, H, W]
        Returns:
            输出特征图 [B, C, H, W]
        """
        # 先应用通道注意力
        x = x * self.channel_attention(x)
        
        # 再应用空间注意力
        x = x * self.spatial_attention(x)
        
        return x


class AttentionResNetBlock(nn.Module):
    """
    带注意力机制的ResNet块
    """
    
    def __init__(self, in_channels, out_channels, stride=1, attention_type='se', reduction=16):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 步长
            attention_type: 注意力类型 ('se', 'cbam', 'none')
            reduction: 注意力模块的降维比例
        """
        super(AttentionResNetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 注意力机制
        if attention_type == 'se':
            self.attention = SEBlock(out_channels, reduction)
        elif attention_type == 'cbam':
            self.attention = CBAM(out_channels, reduction)
        else:
            self.attention = None
        
        # 残差连接
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        """
        Args:
            x: 输入特征图
        Returns:
            输出特征图
        """
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 应用注意力机制
        if self.attention is not None:
            out = self.attention(out)
        
        # 残差连接
        out += identity
        out = self.relu(out)
        
        return out


def add_attention_to_resnet(model, attention_type='se', reduction=16):
    """
    为现有的ResNet模型添加注意力机制
    
    Args:
        model: 预训练的ResNet模型
        attention_type: 注意力类型 ('se', 'cbam')
        reduction: 降维比例
    
    Returns:
        添加了注意力机制的模型
    """
    def add_attention_to_layer(layer, attention_type, reduction):
        """为单个层添加注意力机制"""
        if hasattr(layer, 'conv2') and hasattr(layer, 'bn2'):
            # 这是一个ResNet块
            channels = layer.bn2.num_features
            
            if attention_type == 'se':
                attention = SEBlock(channels, reduction)
            elif attention_type == 'cbam':
                attention = CBAM(channels, reduction)
            else:
                return layer
            
            # 创建新的forward方法
            original_forward = layer.forward
            
            def new_forward(x):
                out = original_forward(x)
                out = attention(out)
                return out
            
            layer.forward = new_forward
            layer.attention = attention
        
        return layer
    
    # 为ResNet的各个层添加注意力机制
    if hasattr(model, 'layer1'):
        for block in model.layer1:
            add_attention_to_layer(block, attention_type, reduction)
    
    if hasattr(model, 'layer2'):
        for block in model.layer2:
            add_attention_to_layer(block, attention_type, reduction)
    
    if hasattr(model, 'layer3'):
        for block in model.layer3:
            add_attention_to_layer(block, attention_type, reduction)
    
    if hasattr(model, 'layer4'):
        for block in model.layer4:
            add_attention_to_layer(block, attention_type, reduction)
    
    return model 