#!/usr/bin/env python3
"""
Generate comparative visualizations of the three models for each disease class
"""

import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'  # Use non-Chinese font

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def create_model_comparison(class_name, input_dir, output_path):
    """Create a comparative image with three models' Grad-CAM visualizations for a single disease class"""
    # Load images
    baseline_path = input_dir / f"baseline_{class_name}.png"
    senet_path = input_dir / f"senet_{class_name}.png"
    cbam_path = input_dir / f"cbam_{class_name}.png"
    
    baseline_img = mpimg.imread(baseline_path)
    senet_img = mpimg.imread(senet_path)
    cbam_img = mpimg.imread(cbam_path)
    
    # Create figure for model comparison
    fig, axes = plt.subplots(3, 1, figsize=(15, 18))
    
    # Display images with titles
    axes[0].imshow(baseline_img)
    axes[0].set_title("Baseline ResNet-50 Model", fontsize=16)
    axes[0].axis('off')
    
    axes[1].imshow(senet_img)
    axes[1].set_title("SE-ResNet-50 Model", fontsize=16)
    axes[1].axis('off')
    
    axes[2].imshow(cbam_img)
    axes[2].set_title("CBAM-ResNet-50 Model", fontsize=16)
    axes[2].axis('off')
    
    # Format class name for title
    formatted_class = class_name.replace('_', ' ').title()
    
    # Set overall title
    fig.suptitle(f'Grad-CAM Comparison: {formatted_class} Sample', fontsize=18)
    
    # Save the figure
    print(f"Saving comparison to: {output_path}")
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison saved to: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Create model comparison visualizations')
    parser.add_argument('--input_dir', type=str, default='outputs/visualization/model_gradcams', help='Directory containing individual Grad-CAM visualizations')
    parser.add_argument('--output_dir', type=str, default='outputs/visualization/model_comparisons', help='Output directory for comparison images')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comparisons for each class
    input_dir = Path(args.input_dir)
    
    classes = ['bacterial_spot', 'target_spot', 'healthy']
    
    for class_name in classes:
        output_path = output_dir / f"{class_name}_model_comparison.png"
        create_model_comparison(class_name, input_dir, output_path)
    
    # Create a comparison summary with all disease classes
    fig, axes = plt.subplots(3, 3, figsize=(20, 20))
    
    for i, class_name in enumerate(classes):
        baseline_path = input_dir / f"baseline_{class_name}.png"
        senet_path = input_dir / f"senet_{class_name}.png"
        cbam_path = input_dir / f"cbam_{class_name}.png"
        
        baseline_img = mpimg.imread(baseline_path)
        senet_img = mpimg.imread(senet_path)
        cbam_img = mpimg.imread(cbam_path)
        
        # Format class name for titles
        formatted_class = class_name.replace('_', ' ').title()
        
        # Row 1: Baseline
        axes[0, i].imshow(baseline_img)
        axes[0, i].set_title(f"Baseline: {formatted_class}", fontsize=14)
        axes[0, i].axis('off')
        
        # Row 2: SE-Net
        axes[1, i].imshow(senet_img)
        axes[1, i].set_title(f"SE-Net: {formatted_class}", fontsize=14)
        axes[1, i].axis('off')
        
        # Row 3: CBAM
        axes[2, i].imshow(cbam_img)
        axes[2, i].set_title(f"CBAM: {formatted_class}", fontsize=14)
        axes[2, i].axis('off')
    
    # Add row titles
    plt.figtext(0.01, 0.835, "Baseline\nResNet-50", fontsize=16, ha='left', va='center')
    plt.figtext(0.01, 0.5, "SE-ResNet-50", fontsize=16, ha='left', va='center')
    plt.figtext(0.01, 0.165, "CBAM-ResNet-50", fontsize=16, ha='left', va='center')
    
    # Set overall title
    fig.suptitle('Comparative Analysis of Attention Mechanisms across Disease Classes', fontsize=20)
    
    # Save the figure
    summary_path = output_dir / "complete_model_comparison.png"
    plt.tight_layout(rect=[0.03, 0, 1, 0.97])  # Adjust for suptitle and row titles
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Complete comparison saved to: {summary_path}")

if __name__ == "__main__":
    main() 