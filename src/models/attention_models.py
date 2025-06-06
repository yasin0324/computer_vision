"""
带注意力机制的完整模型定义
基于改进的基线模型添加SE-Net和CBAM注意力机制
"""

import torch
import torch.nn as nn
from torchvision import models
from .attention import SEBlock, CBAM


def _add_se_to_resnet50(model, reduction=16):
    """为ResNet50的每个Bottleneck块添加SE注意力"""
    
    def add_se_to_bottleneck(bottleneck, reduction):
        """为单个Bottleneck块添加SE注意力"""
        # 获取输出通道数
        out_channels = bottleneck.conv3.out_channels
        
        # 创建SE模块
        se_module = SEBlock(out_channels, reduction)
        
        def se_forward(x):
            # 执行原始的Bottleneck forward
            residual = x
            
            out = bottleneck.conv1(x)
            out = bottleneck.bn1(out)
            out = bottleneck.relu(out)
            
            out = bottleneck.conv2(out)
            out = bottleneck.bn2(out)
            out = bottleneck.relu(out)
            
            out = bottleneck.conv3(out)
            out = bottleneck.bn3(out)
            
            # 应用SE注意力
            out = se_module(out)
            
            if bottleneck.downsample is not None:
                residual = bottleneck.downsample(x)
            
            out += residual
            out = bottleneck.relu(out)
            
            return out
        
        # 替换forward方法
        bottleneck.forward = se_forward
        bottleneck.se = se_module
        
        return bottleneck
    
    # 为各层的Bottleneck块添加SE注意力
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        if hasattr(model, layer_name):
            layer = getattr(model, layer_name)
            for i, bottleneck in enumerate(layer):
                layer[i] = add_se_to_bottleneck(bottleneck, reduction)
    
    return model


def _add_cbam_to_resnet50(model, reduction=16):
    """为ResNet50的每个Bottleneck块添加CBAM注意力"""
    
    def add_cbam_to_bottleneck(bottleneck, reduction):
        """为单个Bottleneck块添加CBAM注意力"""
        # 获取输出通道数
        out_channels = bottleneck.conv3.out_channels
        
        # 创建CBAM模块
        cbam_module = CBAM(out_channels, reduction)
        
        def cbam_forward(x):
            # 执行原始的Bottleneck forward
            residual = x
            
            out = bottleneck.conv1(x)
            out = bottleneck.bn1(out)
            out = bottleneck.relu(out)
            
            out = bottleneck.conv2(out)
            out = bottleneck.bn2(out)
            out = bottleneck.relu(out)
            
            out = bottleneck.conv3(out)
            out = bottleneck.bn3(out)
            
            # 应用CBAM注意力
            out = cbam_module(out)
            
            if bottleneck.downsample is not None:
                residual = bottleneck.downsample(x)
            
            out += residual
            out = bottleneck.relu(out)
            
            return out
        
        # 替换forward方法
        bottleneck.forward = cbam_forward
        bottleneck.cbam = cbam_module
        
        return bottleneck
    
    # 为各层的Bottleneck块添加CBAM注意力
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        if hasattr(model, layer_name):
            layer = getattr(model, layer_name)
            for i, bottleneck in enumerate(layer):
                layer[i] = add_cbam_to_bottleneck(bottleneck, reduction)
    
    return model


def create_resnet_with_attention(
    num_classes: int = 4,
    attention_type: str = 'se',
    reduction: int = 16,
    dropout_rate: float = 0.7,
    pretrained: bool = True,
    freeze_backbone: bool = False
):
    """
    创建带注意力机制的ResNet50模型
    
    Args:
        num_classes: 分类数量
        attention_type: 注意力类型 ('se', 'cbam', 'none')
        reduction: 注意力模块的降维比例
        dropout_rate: Dropout率
        pretrained: 是否使用预训练权重
        freeze_backbone: 是否冻结骨干网络
    
    Returns:
        带注意力机制的模型
    """
    # 创建基础ResNet50模型
    model = models.resnet50(pretrained=pretrained)
    
    # 如果需要添加注意力机制，我们需要重新构建模型
    if attention_type == 'se':
        # 为每个Bottleneck块添加SE注意力
        model = _add_se_to_resnet50(model, reduction)
    elif attention_type == 'cbam':
        # 为每个Bottleneck块添加CBAM注意力
        model = _add_cbam_to_resnet50(model, reduction)
    
    # 冻结骨干网络（如果需要）
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # 替换分类头
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(num_features, num_classes)
    )
    
    # 确保分类头的参数可训练
    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model


class ResNetSE(nn.Module):
    """
    ResNet50 + SE-Net 注意力机制
    """
    
    def __init__(self, num_classes=4, reduction=16, dropout_rate=0.7, pretrained=True):
        super(ResNetSE, self).__init__()
        self.model = create_resnet_with_attention(
            num_classes=num_classes,
            attention_type='se',
            reduction=reduction,
            dropout_rate=dropout_rate,
            pretrained=pretrained
        )
    
    def forward(self, x):
        return self.model(x)


class ResNetCBAM(nn.Module):
    """
    ResNet50 + CBAM 注意力机制
    """
    
    def __init__(self, num_classes=4, reduction=16, dropout_rate=0.7, pretrained=True):
        super(ResNetCBAM, self).__init__()
        self.model = create_resnet_with_attention(
            num_classes=num_classes,
            attention_type='cbam',
            reduction=reduction,
            dropout_rate=dropout_rate,
            pretrained=pretrained
        )
    
    def forward(self, x):
        return self.model(x)


class ResNetDualAttention(nn.Module):
    """
    ResNet50 + 双重注意力机制 (SE + CBAM)
    """
    
    def __init__(self, num_classes=4, reduction=16, dropout_rate=0.7, pretrained=True):
        super(ResNetDualAttention, self).__init__()
        
        # 创建基础模型
        base_model = models.resnet50(pretrained=pretrained)
        
        # 先添加SE注意力
        base_model = _add_se_to_resnet50(base_model, reduction)
        
        # 再添加CBAM注意力（需要特殊处理）
        self._add_cbam_to_se_model(base_model, reduction)
        
        # 替换分类头
        num_features = base_model.fc.in_features
        base_model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )
        
        self.model = base_model
    
    def _add_cbam_to_se_model(self, model, reduction):
        """为已有SE注意力的模型添加CBAM"""
        def add_cbam_to_layer(layer, reduction):
            if hasattr(layer, 'se') and hasattr(layer, 'conv3'):
                # 获取通道数
                channels = layer.conv3.out_channels
                
                # 创建CBAM模块
                cbam = CBAM(channels, reduction)
                
                # 保存原始的forward方法
                original_forward = layer.forward
                
                def new_forward(x):
                    # 先执行原始forward（包含SE注意力）
                    out = original_forward(x)
                    # 再应用CBAM注意力
                    out = cbam(out)
                    return out
                
                layer.forward = new_forward
                layer.cbam = cbam
        
        # 为各层添加CBAM
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            if hasattr(model, layer_name):
                layer = getattr(model, layer_name)
                for block in layer:
                    add_cbam_to_layer(block, reduction)
    
    def forward(self, x):
        return self.model(x)


def load_baseline_and_add_attention(
    baseline_checkpoint_path: str,
    attention_type: str = 'se',
    reduction: int = 16
):
    """
    加载基线模型检查点并添加注意力机制
    
    Args:
        baseline_checkpoint_path: 基线模型检查点路径
        attention_type: 注意力类型
        reduction: 降维比例
    
    Returns:
        带注意力机制的模型
    """
    # 首先创建一个基线模型来获取权重
    baseline_model = models.resnet50(pretrained=False)
    baseline_model.fc = nn.Sequential(
        nn.Dropout(0.7),
        nn.Linear(baseline_model.fc.in_features, 4)
    )
    
    # 加载基线模型的权重
    checkpoint = torch.load(baseline_checkpoint_path, map_location='cpu')
    
    # 提取模型状态字典
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 加载基线权重
    baseline_model.load_state_dict(state_dict)
    
    # 创建带注意力的新模型
    attention_model = create_resnet_with_attention(
        attention_type=attention_type,
        reduction=reduction,
        pretrained=False  # 不使用预训练，因为要从基线模型初始化
    )
    
    # 将基线模型的权重复制到注意力模型（除了注意力模块）
    attention_state_dict = attention_model.state_dict()
    baseline_state_dict = baseline_model.state_dict()
    
    for key in baseline_state_dict:
        if key in attention_state_dict:
            attention_state_dict[key] = baseline_state_dict[key]
    
    attention_model.load_state_dict(attention_state_dict, strict=False)
    
    return attention_model


def get_model_info(model):
    """
    获取模型信息
    
    Args:
        model: 模型
    
    Returns:
        模型信息字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 统计注意力模块数量
    se_blocks = 0
    cbam_blocks = 0
    
    for name, module in model.named_modules():
        if hasattr(module, 'se'):
            se_blocks += 1
        if hasattr(module, 'cbam'):
            cbam_blocks += 1
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'se_blocks': se_blocks,
        'cbam_blocks': cbam_blocks,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # 假设float32
    } 