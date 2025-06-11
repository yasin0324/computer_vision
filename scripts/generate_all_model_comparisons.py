#!/usr/bin/env python3
"""
Generate comparative Grad-CAM heatmaps for all three disease classes
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def main():
    parser = argparse.ArgumentParser(description='Generate comparative Grad-CAM heatmaps for all disease classes')
    parser.add_argument('--baseline_model', type=str, default='outputs/models/resnet50_baseline_improved/best_checkpoint_epoch_25.pth', help='Path to baseline model')
    parser.add_argument('--senet_model', type=str, default='outputs/models/resnet50_se_net/best_checkpoint_epoch_17.pth', help='Path to SE-Net model')
    parser.add_argument('--cbam_model', type=str, default='outputs/models/resnet50_cbam/best_checkpoint_epoch_6.pth', help='Path to CBAM model')
    parser.add_argument('--test_csv', type=str, default='data/processed/test_split.csv', help='Path to test CSV file')
    parser.add_argument('--output_dir', type=str, default='outputs/visualization/model_comparison', help='Output directory')
    parser.add_argument('--layer', type=str, default='layer4', help='Target layer for feature extraction')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate comparisons for all three classes
    classes = ['bacterial_spot', 'target_spot', 'healthy']
    
    for class_name in classes:
        print(f"\n=== Processing class: {class_name} ===")
        output_path = output_dir / f"model_comparison_{class_name}.png"
        
        # Run the model comparison script
        cmd = f"python scripts/generate_model_comparison_gradcams.py" \
              f" --baseline_model \"{args.baseline_model}\"" \
              f" --senet_model \"{args.senet_model}\"" \
              f" --cbam_model \"{args.cbam_model}\"" \
              f" --test_csv \"{args.test_csv}\"" \
              f" --class_name \"{class_name}\"" \
              f" --output_path \"{output_path}\"" \
              f" --layer \"{args.layer}\""
        
        print(f"Running command: {cmd}")
        os.system(cmd)
        
        print(f"Saved comparison to: {output_path}")

if __name__ == "__main__":
    main() 