#!/usr/bin/env python3
"""
Generate individual Grad-CAM heatmaps for different models and samples
"""

import sys
import torch
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import cv2
import matplotlib
import os
import torchvision.models as models
import pandas as pd

# Set font to avoid Chinese characters issue
matplotlib.rcParams['font.family'] = 'Arial'

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.attention_models import ResNetSE, ResNetCBAM
from src.visualization.grad_cam import GradCAM
from src.config.config import Config

def get_sample_image(class_name, image_path=None):
    """Get a sample image for the specified class"""
    if image_path:
        return image_path
    
    # Use a predefined image path for each class if none provided
    class_paths = {
        'bacterial_spot': 'data/raw/PlantVillage/Tomato_Bacterial_spot/f666c0a8-9564-43af-aa93-14192a43816c___GCREC_Bact.Sp 5673.JPG',
        'target_spot': 'data/raw/PlantVillage/Tomato__Target_Spot/a489da29-b369-4bae-a001-f9fe8f2348cc___Com.G_TgS_FL 0735.JPG',
        'healthy': 'data/raw/PlantVillage/Tomato_healthy/74579097-2dd7-4d15-8f73-f09e7d919b51___RS_HL 9773.JPG'
    }
    
    return class_paths.get(class_name, class_paths['bacterial_spot'])

def generate_single_gradcam(model_type, model_path, image_path, output_path, layer_name="layer4"):
    """Generate Grad-CAM heatmap for a single model and image"""
    # Load configuration
    config = Config()
    device = torch.device('cpu')  # Use CPU for compatibility
    
    # Load model based on type
    if model_type == 'baseline':
        # Create a pre-trained ResNet-50 model
        model = models.resnet50(pretrained=True)
        # Replace the fully connected layer
        model.fc = torch.nn.Linear(model.fc.in_features, config.NUM_CLASSES)
    elif model_type == 'senet':
        model = ResNetSE(
            num_classes=config.NUM_CLASSES,
            reduction=16,
            dropout_rate=0.7
        )
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        model.model.load_state_dict(checkpoint['model_state_dict'])
    elif model_type == 'cbam':
        model = ResNetCBAM(
            num_classes=config.NUM_CLASSES,
            reduction=16,
            dropout_rate=0.7
        )
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        model.model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.to(device)
    model.eval()
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    img = Image.open(image_path)
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # Create GradCAM object
    grad_cam = GradCAM(model, layer_name, device=device)
    
    # Forward pass
    output = model(input_tensor)
    pred_class = torch.argmax(output, 1).item()
    
    # Get class name
    class_names = list(config.TARGET_CLASSES.values())
    pred_label = class_names[pred_class]
    
    # Generate CAM
    cam, _ = grad_cam.generate_cam(input_tensor, pred_class)
    
    # Resize CAM to original image size
    cam_resized = cv2.resize(cam, (config.INPUT_SIZE, config.INPUT_SIZE))
    
    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Get original image as numpy array
    img_array = np.array(img.resize((config.INPUT_SIZE, config.INPUT_SIZE)))
    
    # Superimposed image
    superimposed = heatmap * 0.4 + img_array * 0.6
    superimposed = superimposed.astype(np.uint8)
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Extract class name from path for title
    class_name = "Unknown"
    for c in config.TARGET_CLASSES.values():
        if c.lower() in image_path.lower():
            class_name = c
            break
    
    # Set title
    fig.suptitle(f'{model_type.upper()} Model: {class_name} Sample (Prediction: {pred_label})', fontsize=16)
    
    # Original image
    axes[0].imshow(img_array)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis('off')
    
    # Superimposed image
    axes[2].imshow(superimposed)
    axes[2].set_title("Superimposed")
    axes[2].axis('off')
    
    # Save the figure
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Grad-CAM visualization saved to: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Generate Grad-CAM heatmaps for different models and samples')
    parser.add_argument('--baseline_model', type=str, default='outputs/models/resnet50_baseline_improved/best_checkpoint_epoch_25.pth', help='Path to baseline model')
    parser.add_argument('--senet_model', type=str, default='outputs/models/resnet50_se_net/best_checkpoint_epoch_17.pth', help='Path to SE-Net model')
    parser.add_argument('--cbam_model', type=str, default='outputs/models/resnet50_cbam/best_checkpoint_epoch_6.pth', help='Path to CBAM model')
    parser.add_argument('--output_dir', type=str, default='outputs/visualization/model_gradcams', help='Output directory')
    parser.add_argument('--layer', type=str, default='layer4', help='Target layer for feature extraction')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate Grad-CAMs for all models and classes
    models = [
        ('baseline', args.baseline_model),
        ('senet', args.senet_model),
        ('cbam', args.cbam_model)
    ]
    
    classes = ['bacterial_spot', 'target_spot', 'healthy']
    
    for model_type, model_path in models:
        for class_name in classes:
            print(f"\n=== Processing {model_type} model for {class_name} ===")
            
            # Get sample image
            image_path = get_sample_image(class_name)
            image_path = image_path.replace('\\', '/')
            
            # Generate output path
            output_path = output_dir / f"{model_type}_{class_name}.png"
            
            # Generate Grad-CAM
            generate_single_gradcam(
                model_type,
                model_path,
                image_path,
                output_path,
                args.layer
            )
    
    print("\nAll Grad-CAM visualizations generated successfully.")

if __name__ == "__main__":
    main() 