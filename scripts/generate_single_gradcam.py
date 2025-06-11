#!/usr/bin/env python3
"""
Generate Grad-CAM heatmap for SE-Net model
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
matplotlib.rcParams['font.family'] = 'Arial'  # Use non-Chinese font

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.attention_models import ResNetSE
from src.visualization.grad_cam import GradCAM
from src.config.config import Config

def generate_gradcam(model_path, image_path, output_path, layer_name="layer4"):
    """Generate Grad-CAM heatmap for a single image"""
    # Load configuration
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model: {model_path}")
    model = ResNetSE(
        num_classes=config.NUM_CLASSES,
        reduction=16,
        dropout_rate=0.7
    )
    checkpoint = torch.load(model_path, map_location=device)
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load image
    print(f"Loading image: {image_path}")
    img = Image.open(image_path)
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # Create GradCAM object
    print(f"Creating GradCAM object, target layer: {layer_name}")
    grad_cam = GradCAM(model, layer_name, device=device)
    
    # Forward pass
    print(f"Performing forward inference...")
    output = model(input_tensor)
    pred_class = torch.argmax(output, 1).item()
    
    # Get class name
    class_names = list(config.TARGET_CLASSES.values())
    pred_label = class_names[pred_class]
    print(f"Predicted class: {pred_label}")
    
    # Generate CAM
    print(f"Generating CAM...")
    cam, _ = grad_cam.generate_cam(input_tensor, pred_class)
    
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
    
    # Save visualization
    print(f"Saving visualization to: {output_path}")
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_array)
    plt.title(f"Original Image\nPrediction: {pred_label}")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(superimposed)
    plt.title("Attention Focus")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Grad-CAM visualization saved to: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Generate Grad-CAM heatmap for SE-Net model')
    parser.add_argument('--model_path', type=str, default='outputs/models/resnet50_se_net/best_checkpoint_epoch_17.pth', help='Path to SE-Net model')
    parser.add_argument('--image_path', type=str, required=True, help='Input image path')
    parser.add_argument('--output_path', type=str, required=True, help='Output image path')
    parser.add_argument('--layer', type=str, default='layer4', help='Target layer for feature extraction')
    
    args = parser.parse_args()
    
    generate_gradcam(args.model_path, args.image_path, args.output_path, args.layer)

if __name__ == "__main__":
    main() 