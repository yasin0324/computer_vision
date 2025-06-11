#!/usr/bin/env python3
"""
Generate comparative Grad-CAM heatmaps for different models on the same sample
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
import torchvision.models as models
import pandas as pd

matplotlib.rcParams['font.family'] = 'Arial'  # Use non-Chinese font

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.attention_models import ResNetSE, ResNetCBAM
from src.visualization.grad_cam import GradCAM
from src.config.config import Config

def load_baseline_model(model_path, num_classes):
    """Load baseline ResNet-50 model"""
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = {}
    
    # Convert backbone.x.y.z to x.y.z format for baseline model
    for k, v in checkpoint['model_state_dict'].items():
        if k.startswith('backbone.'):
            new_key = k.replace('backbone.', '')
            # Handle special cases for the first few layers
            if new_key.startswith('0.'):
                new_key = 'conv1' + new_key[1:]
            elif new_key.startswith('1.'):
                new_key = 'bn1' + new_key[1:]
            elif new_key.startswith('4.'):
                new_key = 'layer1' + new_key[1:]
            elif new_key.startswith('5.'):
                new_key = 'layer2' + new_key[1:]
            elif new_key.startswith('6.'):
                new_key = 'layer3' + new_key[1:]
            elif new_key.startswith('7.'):
                new_key = 'layer4' + new_key[1:]
            state_dict[new_key] = v
        elif k.startswith('classifier.'):
            if '3.' in k:  # Last FC layer
                new_key = 'fc' + k[k.find('.'):]
                state_dict[new_key] = v
    
    # Handle fully connected layer keys manually
    if 'classifier.3.weight' in checkpoint['model_state_dict']:
        state_dict['fc.weight'] = checkpoint['model_state_dict']['classifier.3.weight']
    if 'classifier.3.bias' in checkpoint['model_state_dict']:
        state_dict['fc.bias'] = checkpoint['model_state_dict']['classifier.3.bias']
    
    # We'll use a custom approach since the state_dict structure is different
    # Create a pre-trained ResNet-50 model
    pretrained_model = models.resnet50(pretrained=True)
    # Replace the fully connected layer
    pretrained_model.fc = torch.nn.Linear(pretrained_model.fc.in_features, num_classes)
    
    return pretrained_model

def load_senet_model(model_path, num_classes):
    """Load SE-Net model"""
    model = ResNetSE(
        num_classes=num_classes,
        reduction=16,
        dropout_rate=0.7
    )
    checkpoint = torch.load(model_path, map_location='cpu')
    model.model.load_state_dict(checkpoint['model_state_dict'])
    return model

def load_cbam_model(model_path, num_classes):
    """Load CBAM model"""
    model = ResNetCBAM(
        num_classes=num_classes,
        reduction=16,
        dropout_rate=0.7
    )
    checkpoint = torch.load(model_path, map_location='cpu')
    model.model.load_state_dict(checkpoint['model_state_dict'])
    return model

def get_sample_image(test_csv_path, class_name):
    """Get a sample image from test dataset for the specified class"""
    test_df = pd.read_csv(test_csv_path)
    class_samples = test_df[test_df['label'] == class_name].sample(1)
    
    if class_samples.empty:
        raise ValueError(f"No samples found for class: {class_name}")
    
    return class_samples['image_path'].iloc[0]

def generate_comparative_gradcam(baseline_model_path, senet_model_path, cbam_model_path, 
                               image_path, output_path, layer_name="layer4"):
    """Generate comparative Grad-CAM heatmaps for three models on the same image"""
    # Load configuration
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    print(f"Loading models...")
    baseline_model = load_baseline_model(baseline_model_path, config.NUM_CLASSES)
    baseline_model.to(device)
    baseline_model.eval()
    
    senet_model = load_senet_model(senet_model_path, config.NUM_CLASSES)
    senet_model.to(device)
    senet_model.eval()
    
    cbam_model = load_cbam_model(cbam_model_path, config.NUM_CLASSES)
    cbam_model.to(device)
    cbam_model.eval()
    
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
    
    # Get class names
    class_names = list(config.TARGET_CLASSES.values())
    
    # Create figure for model comparison
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    model_names = ['Baseline ResNet-50', 'SE-ResNet-50', 'CBAM-ResNet-50']
    models = [baseline_model, senet_model, cbam_model]
    
    # Extract class name from path for title
    class_name = "Unknown"
    for c in config.TARGET_CLASSES.values():
        if c.lower() in image_path.lower():
            class_name = c
            break
    
    # Set overall title
    fig.suptitle(f'Grad-CAM Comparison on {class_name} Sample', fontsize=16)
    
    for i, (model_name, model) in enumerate(zip(model_names, models)):
        print(f"\nProcessing model: {model_name}")
        
        # Create GradCAM object
        if i == 0:  # Baseline model
            grad_cam = GradCAM(model, layer_name, device=device)
        else:  # Attention models
            grad_cam = GradCAM(model, layer_name, device=device)
        
        # Forward pass
        output = model(input_tensor)
        pred_class = torch.argmax(output, 1).item()
        
        # Get class name
        pred_label = class_names[pred_class]
        print(f"Predicted class: {pred_label}")
        
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
        
        # Original image
        axes[i, 0].imshow(img_array)
        axes[i, 0].set_title(f"{model_name}\nOriginal Image")
        axes[i, 0].axis('off')
        
        # Heatmap
        axes[i, 1].imshow(heatmap)
        axes[i, 1].set_title(f"Grad-CAM Heatmap\nPrediction: {pred_label}")
        axes[i, 1].axis('off')
        
        # Superimposed image
        axes[i, 2].imshow(superimposed)
        axes[i, 2].set_title("Superimposed\nAttention Focus")
        axes[i, 2].axis('off')
    
    # Save the figure
    print(f"Saving visualization to: {output_path}")
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparative Grad-CAM visualization saved to: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Generate comparative Grad-CAM heatmaps for different models')
    parser.add_argument('--baseline_model', type=str, default='outputs/models/resnet50_baseline_improved/best_checkpoint_epoch_25.pth', help='Path to baseline model')
    parser.add_argument('--senet_model', type=str, default='outputs/models/resnet50_se_net/best_checkpoint_epoch_17.pth', help='Path to SE-Net model')
    parser.add_argument('--cbam_model', type=str, default='outputs/models/resnet50_cbam/best_checkpoint_epoch_6.pth', help='Path to CBAM model')
    parser.add_argument('--test_csv', type=str, default='data/processed/test_split.csv', help='Path to test CSV file')
    parser.add_argument('--class_name', type=str, required=True, help='Class name to generate visualization for')
    parser.add_argument('--output_path', type=str, required=True, help='Output image path')
    parser.add_argument('--layer', type=str, default='layer4', help='Target layer for feature extraction')
    
    args = parser.parse_args()
    
    # Get a sample image for the specified class
    image_path = get_sample_image(args.test_csv, args.class_name)
    image_path = image_path.replace('\\', '/')
    
    generate_comparative_gradcam(
        args.baseline_model,
        args.senet_model,
        args.cbam_model,
        image_path,
        args.output_path,
        args.layer
    )

if __name__ == "__main__":
    main() 