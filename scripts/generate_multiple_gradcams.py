#!/usr/bin/env python3
"""
Generate Grad-CAM heatmaps for multiple disease samples
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import random

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def get_sample_images(test_csv_path, classes=None, samples_per_class=1):
    """Get sample images from test dataset"""
    test_df = pd.read_csv(test_csv_path)
    
    if classes is None:
        classes = test_df['label'].unique().tolist()
    
    samples = []
    for class_name in classes:
        class_samples = test_df[test_df['label'] == class_name].sample(min(samples_per_class, len(test_df[test_df['label'] == class_name])))
        samples.extend(class_samples['image_path'].tolist())
    
    return samples

def main():
    parser = argparse.ArgumentParser(description='Generate Grad-CAM heatmaps for multiple samples')
    parser.add_argument('--model_path', type=str, default='outputs/models/resnet50_se_net/best_checkpoint_epoch_17.pth', help='Path to SE-Net model')
    parser.add_argument('--test_csv', type=str, default='data/processed/test_split.csv', help='Path to test CSV file')
    parser.add_argument('--output_dir', type=str, default='outputs/visualization/senet_gradcam', help='Output directory')
    parser.add_argument('--layer', type=str, default='layer4', help='Target layer for feature extraction')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get samples for bacterial_spot, target_spot, and healthy
    target_classes = ['bacterial_spot', 'target_spot', 'healthy']
    samples = get_sample_images(args.test_csv, classes=target_classes, samples_per_class=1)
    
    for i, sample_path in enumerate(samples):
        print(f"Processing sample {i+1}/{len(samples)}: {sample_path}")
        
        # Convert Windows path to proper format
        sample_path = sample_path.replace('\\', '/')
        
        # Extract class name from the path
        class_name = next(c for c in target_classes if c in sample_path.lower())
        output_path = output_dir / f"senet_gradcam_{class_name}.png"
        
        # Run the generate_single_gradcam.py script
        cmd = f"python scripts/generate_single_gradcam.py" \
              f" --model_path \"{args.model_path}\"" \
              f" --image_path \"{sample_path}\"" \
              f" --output_path \"{output_path}\"" \
              f" --layer \"{args.layer}\""
        
        print(f"Running command: {cmd}")
        os.system(cmd)
        
        print(f"Saved output to: {output_path}")

if __name__ == "__main__":
    main() 