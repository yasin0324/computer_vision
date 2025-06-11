#!/usr/bin/env python3
"""
Generate model comparison image from existing individual model Grad-CAM images
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

def combine_images(se_image_path, output_path, title="Grad-CAM Comparison"):
    """Create a composite image with SE-Net Grad-CAM visualization"""
    # Load SE-Net image
    se_img = mpimg.imread(se_image_path)
    
    # Create figure for model comparison
    plt.figure(figsize=(12, 5))
    
    # Set title
    plt.title(title, fontsize=16)
    plt.imshow(se_img)
    plt.axis('off')
    
    # Save the figure
    print(f"Saving combined visualization to: {output_path}")
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined visualization saved to: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Create model comparison visualization')
    parser.add_argument('--bacterial_spot_image', type=str, default='outputs/visualization/senet_gradcam/senet_gradcam_bacterial_spot.png', help='Path to bacterial spot image')
    parser.add_argument('--target_spot_image', type=str, default='outputs/visualization/senet_gradcam/senet_gradcam_target_spot.png', help='Path to target spot image')
    parser.add_argument('--healthy_image', type=str, default='outputs/visualization/senet_gradcam/senet_gradcam_healthy.png', help='Path to healthy image')
    parser.add_argument('--output_dir', type=str, default='outputs/visualization/model_comparison', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate comparisons for all three classes
    combine_images(
        args.bacterial_spot_image,
        output_dir / "bacterial_spot_comparison.png",
        title="SE-Net Grad-CAM: Bacterial Spot Sample"
    )
    
    combine_images(
        args.target_spot_image,
        output_dir / "target_spot_comparison.png",
        title="SE-Net Grad-CAM: Target Spot Sample"
    )
    
    combine_images(
        args.healthy_image,
        output_dir / "healthy_comparison.png",
        title="SE-Net Grad-CAM: Healthy Sample"
    )
    
    # Create a 3-row figure with all three model comparisons
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Load all images
    bacterial_img = mpimg.imread(args.bacterial_spot_image)
    target_img = mpimg.imread(args.target_spot_image)
    healthy_img = mpimg.imread(args.healthy_image)
    
    # Display images
    axes[0].imshow(bacterial_img)
    axes[0].set_title("SE-Net Grad-CAM: Bacterial Spot Sample", fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(target_img)
    axes[1].set_title("SE-Net Grad-CAM: Target Spot Sample", fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(healthy_img)
    axes[2].set_title("SE-Net Grad-CAM: Healthy Sample", fontsize=14)
    axes[2].axis('off')
    
    # Save combined figure
    plt.tight_layout()
    plt.savefig(output_dir / "all_samples_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"All samples comparison saved to: {output_dir / 'all_samples_comparison.png'}")

if __name__ == "__main__":
    main() 