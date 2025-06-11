#!/usr/bin/env python3
"""
Generate Grad-CAM heatmaps for different models
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
import traceback

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.attention_models import ResNetSE, ResNetCBAM
import torchvision.models as models
from src.visualization.grad_cam import GradCAM
from src.config.config import Config

def load_model(model_type, model_path, num_classes):
    """Load a specific model based on type"""
    if model_type == 'baseline':
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    elif model_type == 'senet':
        model = ResNetSE(
            num_classes=num_classes,
            reduction=16,
            dropout_rate=0.7
        )
        checkpoint = torch.load(model_path, map_location='cpu')
        model.model.load_state_dict(checkpoint['model_state_dict'])
        return model
    elif model_type == 'cbam':
        model = ResNetCBAM(
            num_classes=num_classes,
            reduction=16,
            dropout_rate=0.7
        )
        checkpoint = torch.load(model_path, map_location='cpu')
        model.model.load_state_dict(checkpoint['model_state_dict'])
        return model
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def generate_gradcam(models_info, image_path, output_path, layer_name="layer4"):
    """Generate Grad-CAM heatmaps for multiple models"""
    try:
        print(f"Starting Grad-CAM heatmap generation...")
        print(f"Image path: {image_path}")
        print(f"Output path: {output_path}")
        
        # Load configuration
        config = Config()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load image
        print(f"Loading image...")
        img = Image.open(image_path)
        input_tensor = transform(img).unsqueeze(0).to(device)
        print(f"Image size: {img.size}, tensor shape: {input_tensor.shape}")
        
        # Get class names
        class_names = list(config.TARGET_CLASSES.values())
        
        # Create figure for multiple models
        fig, axes = plt.subplots(len(models_info), 3, figsize=(15, 5*len(models_info)))
        
        # Process each model
        for i, (model_name, model_info) in enumerate(models_info.items()):
            print(f"\nProcessing model: {model_name}")
            
            # Load model
            model_type = model_info['type']
            model_path = model_info['path']
            print(f"Loading model: {model_path}")
            
            model = load_model(model_type, model_path, config.NUM_CLASSES)
            model.to(device)
            model.eval()
            
            # Create GradCAM object
            print(f"Creating GradCAM object, target layer: {layer_name}")
            
            # For baseline model, use direct layer access
            if model_type == 'baseline':
                grad_cam = GradCAM(model, layer_name, device=device)
            else:
                grad_cam = GradCAM(model, layer_name, device=device)
            
            # Forward pass
            print(f"Performing forward inference...")
            output = model(input_tensor)
            pred_class = torch.argmax(output, 1).item()
            print(f"Predicted class index: {pred_class}")
            
            print(f"Generating CAM...")
            cam, _ = grad_cam.generate_cam(input_tensor, pred_class)
            print(f"CAM shape: {cam.shape}")
            
            # Get class name
            pred_label = class_names[pred_class]
            print(f"Predicted class: {pred_label}")
            
            # Visualization
            print(f"Generating heatmap...")
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
            
            # Plot on the corresponding row
            ax_row = axes[i] if len(models_info) > 1 else axes
            
            # Original image
            ax_row[0].imshow(img_array)
            ax_row[0].set_title(f"Original Image\nModel: {model_name}")
            ax_row[0].axis('off')
            
            # Heatmap
            ax_row[1].imshow(heatmap)
            ax_row[1].set_title(f"Grad-CAM Heatmap\nPrediction: {pred_label}")
            ax_row[1].axis('off')
            
            # Superimposed image
            ax_row[2].imshow(superimposed)
            ax_row[2].set_title(f"Superimposed\nAttention Focus")
            ax_row[2].axis('off')
        
        # Save the figure
        print(f"Saving visualization results...")
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Grad-CAM heatmaps saved to: {output_path}")
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Generate Grad-CAM heatmaps')
    parser.add_argument('--baseline_model', type=str, required=True, help='Path to baseline model')
    parser.add_argument('--senet_model', type=str, required=True, help='Path to SE-Net model')
    parser.add_argument('--cbam_model', type=str, required=True, help='Path to CBAM model')
    parser.add_argument('--image_path', type=str, required=True, help='Input image path')
    parser.add_argument('--output_path', type=str, required=True, help='Output image path')
    parser.add_argument('--layer', type=str, default='layer4', help='Target layer for feature extraction')
    
    args = parser.parse_args()
    
    # Define models information
    models_info = {
        'Baseline': {
            'type': 'baseline',
            'path': args.baseline_model
        },
        'SE-Net': {
            'type': 'senet',
            'path': args.senet_model
        },
        'CBAM': {
            'type': 'cbam',
            'path': args.cbam_model
        }
    }
    
    success = generate_gradcam(models_info, args.image_path, args.output_path, args.layer)
    if success:
        print("Grad-CAM generation successful!")
    else:
        print("Grad-CAM generation failed, please check error messages.")

if __name__ == "__main__":
    main() 