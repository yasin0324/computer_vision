"""
Data transformation utilities for tomato spot disease recognition.
"""

from typing import Any
from torchvision import transforms

def get_data_transforms(input_size: int = 224, augment: bool = True) -> Any:
    """
    Get data transformation pipeline.
    
    Args:
        input_size: Target image size
        augment: Whether to apply data augmentation
        
    Returns:
        Composed transforms
    """
    # Base transforms for validation and testing
    base_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )  # ImageNet normalization
    ])
    
    if not augment:
        return base_transform
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((input_size + 32, input_size + 32)),  # Slightly larger for random crop
        transforms.RandomCrop((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2, 
            contrast=0.2, 
            saturation=0.2, 
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform

def get_test_time_augmentation_transforms(input_size: int = 224) -> list:
    """
    Get test time augmentation transforms.
    
    Args:
        input_size: Target image size
        
    Returns:
        List of transform compositions for TTA
    """
    base_transforms = [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ]
    
    tta_transforms = [
        # Original
        transforms.Compose(base_transforms),
        
        # Horizontal flip
        transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ]),
        
        # Vertical flip
        transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ]),
        
        # Rotation
        transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    ]
    
    return tta_transforms

def denormalize_tensor(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize a tensor for visualization.
    
    Args:
        tensor: Normalized tensor
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Denormalized tensor
    """
    import torch
    
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    return tensor * std + mean 