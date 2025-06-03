"""
Visualization utility functions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot training and validation curves.
    
    Args:
        train_losses: Training losses
        val_losses: Validation losses
        train_accs: Training accuracies (optional)
        val_accs: Validation accuracies (optional)
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2 if train_accs is not None else 1, figsize=(12, 5))
    
    if train_accs is not None:
        # Plot losses
        axes[0].plot(train_losses, label='Training Loss', color='blue')
        axes[0].plot(val_losses, label='Validation Loss', color='red')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot accuracies
        axes[1].plot(train_accs, label='Training Accuracy', color='blue')
        axes[1].plot(val_accs, label='Validation Accuracy', color='red')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    else:
        # Only plot losses
        axes.plot(train_losses, label='Training Loss', color='blue')
        axes.plot(val_losses, label='Validation Loss', color='red')
        axes.set_title('Training and Validation Loss')
        axes.set_xlabel('Epoch')
        axes.set_ylabel('Loss')
        axes.legend()
        axes.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = 'Confusion Matrix',
    save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show() 