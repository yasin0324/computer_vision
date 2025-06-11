#!/usr/bin/env python3
"""
Generate comparative Grad-CAM heatmaps for different disease samples and models
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

def get_sample_images(test_csv_path, num_samples=1, class_name=None):
    """Get sample images from test dataset"""
    test_df = pd.read_csv(test_csv_path)
    
    if class_name:
        # Filter by class
        filtered_df = test_df[test_df['label'] == class_name]
        if filtered_df.empty:
            print(f"No samples found for class: {class_name}")
            return []
        # Randomly select samples
        samples = filtered_df.sample(min(num_samples, len(filtered_df)))
    else:
        # Select one sample from each class
        samples = []
        for label in test_df['label'].unique():
            class_samples = test_df[test_df['label'] == label].sample(min(num_samples, len(test_df[test_df['label'] == label])))
            samples.append(class_samples)
        samples = pd.concat(samples)
    
    return samples['image_path'].tolist()

def main():
    parser = argparse.ArgumentParser(description='Generate comparative Grad-CAM heatmaps')
    parser.add_argument('--baseline_model', type=str, default='outputs/models/resnet50_baseline_improved/best_checkpoint_epoch_25.pth', help='Path to baseline model')
    parser.add_argument('--senet_model', type=str, default='outputs/models/resnet50_se_net/best_checkpoint_epoch_17.pth', help='Path to SE-Net model')
    parser.add_argument('--cbam_model', type=str, default='outputs/models/resnet50_cbam/best_checkpoint_epoch_6.pth', help='Path to CBAM model')
    parser.add_argument('--test_csv', type=str, default='data/processed/test_split.csv', help='Path to test CSV file')
    parser.add_argument('--output_dir', type=str, default='outputs/visualization/comparative_gradcam', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get one sample from each class (we have 4 classes)
    classes = ['bacterial_spot', 'septoria_leaf_spot', 'target_spot', 'healthy']
    
    for i, class_name in enumerate(classes):
        print(f"\nProcessing class: {class_name}")
        samples = get_sample_images(args.test_csv, num_samples=1, class_name=class_name)
        
        if not samples:
            print(f"No samples found for class: {class_name}")
            continue
        
        for j, sample_path in enumerate(samples):
            print(f"Processing sample {j+1}/{len(samples)}: {sample_path}")
            
            # Convert Windows path to proper format
            sample_path = sample_path.replace('\\', '/')
            
            output_path = output_dir / f"{class_name}_sample_{j+1}_comparative.png"
            
            # Run the generate_gradcam.py script
            cmd = f"python scripts/generate_gradcam.py" \
                  f" --baseline_model \"{args.baseline_model}\"" \
                  f" --senet_model \"{args.senet_model}\"" \
                  f" --cbam_model \"{args.cbam_model}\"" \
                  f" --image_path \"{sample_path}\"" \
                  f" --output_path \"{output_path}\""
            
            print(f"Running command: {cmd}")
            os.system(cmd)
            
            print(f"Saved output to: {output_path}")

if __name__ == "__main__":
    main() 