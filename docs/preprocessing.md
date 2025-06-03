# Tomato Spot Disease Fine-grained Recognition - Data Preprocessing

This project implements attention mechanism-based fine-grained recognition research for plant leaf diseases, focusing on tomato spot diseases.

## Data Preprocessing Overview

### Target Classes

We selected the following 4 tomato categories for fine-grained recognition research:

- `bacterial_spot` - Bacterial spot disease
- `septoria_leaf_spot` - Septoria leaf spot
- `target_spot` - Target spot disease
- `healthy` - Healthy control

These categories have visual similarities, making them ideal for testing attention mechanisms in fine-grained recognition.

## File Structure

```
├── data_preprocessing.py      # Main data preprocessing module
├── config.py                 # Experiment configuration file
├── run_preprocessing.py      # Script to run preprocessing
├── validate_preprocessing.py # Validate preprocessing results
├── test_visualization.py    # Test visualization functionality
├── README_preprocessing.md   # This documentation
└── processed_data/          # Processed data directory
    ├── train_split.csv      # Training set split
    ├── val_split.csv        # Validation set split
    ├── test_split.csv       # Test set split
    ├── class_mapping.json   # Class mapping
    └── preprocessing_summary.json # Preprocessing summary
```

## Usage

### 1. Run Data Preprocessing

```bash
python run_preprocessing.py
```

This script will:

- Collect all images from target categories
- Analyze data distribution and quality
- Create train/validation/test data splits (60%/20%/20%)
- Generate data loaders
- Visualize data samples

### 2. Validate Preprocessing Results

```bash
python validate_preprocessing.py
```

This script will:

- Check if all necessary files exist
- Validate data integrity
- Check class distribution consistency
- Test data loading functionality
- Generate validation report

### 3. Test Visualization (Optional)

```bash
python test_visualization.py
```

This script tests the visualization functionality with English text to ensure proper display.

## Data Preprocessing Features

### Data Augmentation Strategy

- **Geometric transformations**: Random horizontal flip, vertical flip, rotation
- **Color transformations**: Brightness, contrast, saturation, hue adjustments
- **Size processing**: Unified resizing to 224x224 pixels

### Data Splitting Strategy

- Uses stratified sampling to ensure consistent class proportions across train/validation/test sets
- Training set: 60%, Validation set: 20%, Test set: 20%
- Fixed random seed ensures reproducible results

### Quality Control

- Automatic detection of corrupted image files
- Image sharpness metric calculation
- Statistics on image dimensions and format distribution

## Configuration Parameters

Main configuration parameters are defined in `config.py`:

```python
# Image processing parameters
INPUT_SIZE = 224        # Input image size
BATCH_SIZE = 32         # Batch size

# Data split ratios
TEST_SIZE = 0.2         # Test set ratio
VAL_SIZE = 0.2          # Validation set ratio

# Data augmentation parameters
AUGMENTATION = {
    'horizontal_flip_prob': 0.5,
    'vertical_flip_prob': 0.3,
    'rotation_degrees': 15,
    'color_jitter': {...}
}
```

## Output Files Description

### CSV Files

- `train_split.csv`: Contains training set image paths and labels
- `val_split.csv`: Contains validation set image paths and labels
- `test_split.csv`: Contains test set image paths and labels

### JSON Files

- `class_mapping.json`: Mapping from class names to numeric labels
- `preprocessing_summary.json`: Statistical summary of preprocessing

### Visualization Files

- `tomato_spot_disease_distribution.png`: Dataset class distribution chart
- `sample_images.png`: Data sample visualization
- `data_split_validation.png`: Data split validation chart

## Language and Display

**Note**: All visualization text has been converted to English to avoid font display issues. The charts will now display properly without requiring Chinese font support.

## Next Steps

After data preprocessing is complete, you can proceed with:

1. Baseline model training (ResNet50 without attention)
2. Attention mechanism integration (SE-Net, CBAM)
3. Model performance evaluation and comparison

## Notes

1. Ensure PlantVillage dataset is correctly downloaded to `data/PlantVillage/` directory
2. Install required dependencies before running: torch, torchvision, PIL, opencv-python, matplotlib, seaborn, pandas, scikit-learn
3. If encountering memory issues, reduce `BATCH_SIZE` or `NUM_WORKERS`
4. Data preprocessing generates visualization charts that require graphical interface support
5. All text in visualizations is now in English to ensure proper display across different systems
