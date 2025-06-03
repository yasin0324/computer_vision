"""
PyTorch Dataset classes for tomato spot disease recognition.
"""

import os
from typing import Dict, Optional, Tuple, Any
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class TomatoSpotDataset(Dataset):
    """
    PyTorch Dataset class for loading tomato spot disease data.
    
    Args:
        dataframe: DataFrame containing image paths and labels
        transform: Optional transform to be applied on images
        label_to_idx: Dictionary mapping label strings to indices
    """
    
    def __init__(
        self, 
        dataframe: pd.DataFrame, 
        transform: Optional[Any] = None, 
        label_to_idx: Optional[Dict[str, int]] = None
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        
        # Create label mapping if not provided
        if label_to_idx is None:
            unique_labels = sorted(dataframe['label'].unique())
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_to_idx = label_to_idx
            
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.dataframe)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image, label, image_path)
        """
        row = self.dataframe.iloc[idx]
        image_path = row['image_path']
        label_str = row['label']
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (256, 256), (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        # Convert label to index
        label = self.label_to_idx[label_str]
        
        return image, label, image_path
    
    def get_class_counts(self) -> Dict[str, int]:
        """Get count of samples for each class."""
        return self.dataframe['label'].value_counts().to_dict()
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced training."""
        class_counts = self.get_class_counts()
        total_samples = len(self.dataframe)
        num_classes = len(class_counts)
        
        weights = []
        for i in range(num_classes):
            label = self.idx_to_label[i]
            weight = total_samples / (num_classes * class_counts[label])
            weights.append(weight)
            
        return torch.FloatTensor(weights)
    
    def get_sample_weights(self) -> torch.Tensor:
        """Calculate sample weights for weighted sampling."""
        class_counts = self.get_class_counts()
        sample_weights = []
        
        for _, row in self.dataframe.iterrows():
            label = row['label']
            weight = 1.0 / class_counts[label]
            sample_weights.append(weight)
            
        return torch.FloatTensor(sample_weights) 