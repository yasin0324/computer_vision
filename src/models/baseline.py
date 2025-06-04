"""
基线模型：ResNet50 用于番茄叶斑病细粒度识别
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class ResNetBaseline(nn.Module):
    """
    基于ResNet50的基线模型
    
    Args:
        num_classes: 分类类别数
        pretrained: 是否使用ImageNet预训练权重
        dropout_rate: Dropout比率
        freeze_backbone: 是否冻结骨干网络
    """
    
    def __init__(
        self, 
        num_classes: int = 4,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = False
    ):
        super(ResNetBaseline, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # 加载预训练的ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # 获取特征维度
        self.feature_dim = self.backbone.fc.in_features
        
        # 移除原始的全连接层
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # 冻结骨干网络（如果需要）
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 自定义分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # 初始化分类头权重
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        """初始化分类头的权重"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像张量 [batch_size, 3, 224, 224]
            
        Returns:
            logits: 分类logits [batch_size, num_classes]
        """
        # 特征提取
        features = self.backbone(x)  # [batch_size, 2048, 7, 7]
        
        # 分类
        logits = self.classifier(features)  # [batch_size, num_classes]
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取特征表示（用于可视化和分析）
        
        Args:
            x: 输入图像张量
            
        Returns:
            features: 特征张量 [batch_size, 2048, 7, 7]
        """
        with torch.no_grad():
            features = self.backbone(x)
        return features
    
    def freeze_backbone(self):
        """冻结骨干网络"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """解冻骨干网络"""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'ResNet50_Baseline',
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'feature_dim': self.feature_dim,
            'dropout_rate': self.dropout_rate
        }


def create_resnet_baseline(
    num_classes: int = 4,
    pretrained: bool = True,
    dropout_rate: float = 0.5,
    freeze_backbone: bool = False
) -> ResNetBaseline:
    """
    创建ResNet50基线模型的便捷函数
    
    Args:
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
        dropout_rate: Dropout比率
        freeze_backbone: 是否冻结骨干网络
        
    Returns:
        model: ResNet50基线模型
    """
    model = ResNetBaseline(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone
    )
    
    return model


if __name__ == "__main__":
    # 测试模型
    model = create_resnet_baseline(num_classes=4)
    
    # 打印模型信息
    info = model.get_model_info()
    print("Model Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 测试前向传播
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    # 测试特征提取
    features = model.get_features(x)
    print(f"Features shape: {features.shape}") 