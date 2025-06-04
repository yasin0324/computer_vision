"""
验证数据预处理结果
"""
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import os
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.data.preprocessing import TomatoSpotDataset, get_data_transforms
from src.config import config
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def validate_data_splits(processed_data_dir=None):
    """验证数据划分结果"""
    if processed_data_dir is None:
        processed_data_dir = config.PROCESSED_DATA_DIR
        
    print("=== Validating Data Preprocessing Results ===\n")
    print(f"Checking data in directory: {processed_data_dir}")
    
    # 1. 检查文件是否存在
    required_files = [
        'train_split.csv', 'val_split.csv', 'test_split.csv',
        'class_mapping.json', 'preprocessing_summary.json'
    ]
    
    print("\n1. Checking required files...")
    missing_files = []
    for file_name in required_files:
        file_path = os.path.join(processed_data_dir, file_name)
        if os.path.exists(file_path):
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name} - File does not exist")
            missing_files.append(file_name)
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        print("Please run data preprocessing first: python scripts/preprocess_data.py")
        return False
    
    # 2. 加载和验证数据
    print("\n2. Validating data integrity...")
    
    train_df = pd.read_csv(f"{processed_data_dir}/train_split.csv")
    val_df = pd.read_csv(f"{processed_data_dir}/val_split.csv")
    test_df = pd.read_csv(f"{processed_data_dir}/test_split.csv")
    
    with open(f"{processed_data_dir}/class_mapping.json", 'r') as f:
        class_mapping = json.load(f)
    
    with open(f"{processed_data_dir}/preprocessing_summary.json", 'r') as f:
        summary = json.load(f)
    
    print(f"Training set: {len(train_df)} images")
    print(f"Validation set: {len(val_df)} images")
    print(f"Test set: {len(test_df)} images")
    print(f"Total: {len(train_df) + len(val_df) + len(test_df)} images")
    
    # 3. 验证类别分布
    print("\n3. Validating class distribution...")
    
    all_labels = list(train_df['label']) + list(val_df['label']) + list(test_df['label'])
    label_counts = Counter(all_labels)
    
    print("Overall class distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} images")
    
    # 4. 验证数据加载
    print("\n4. Validating data loading...")
    
    try:
        # 创建小批量数据加载器进行测试
        transform = get_data_transforms(224, augment=False)
        test_dataset = TomatoSpotDataset(
            train_df.head(10), transform, class_mapping['label_to_idx']
        )
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        
        # 尝试加载一批数据
        for batch_idx, (images, labels, paths) in enumerate(test_loader):
            print(f"✅ Successfully loaded batch {batch_idx + 1}: {images.shape}")
            if batch_idx >= 2:  # 只测试前几个批次
                break
                
    except Exception as e:
        print(f"❌ Data loading failed: {str(e)}")
        return False
    
    # 5. 生成验证报告
    print("\n5. Generating validation report...")
    
    # 可视化类别分布对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    splits = [('Training Set', train_df), ('Validation Set', val_df), ('Test Set', test_df)]
    
    for idx, (split_name, df) in enumerate(splits):
        label_dist = df['label'].value_counts().sort_index()
        
        bars = axes[idx].bar(label_dist.index, label_dist.values, 
                           color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[idx].set_title(f'{split_name}\n({len(df)} images)', fontweight='bold')
        axes[idx].set_xlabel('Disease Category')
        axes[idx].set_ylabel('Number of Images')
        axes[idx].tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, count in zip(bars, label_dist.values):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                         str(count), ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Data Split Validation - Class Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{processed_data_dir}/data_split_validation.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. 计算分布一致性
    print("\n6. Distribution consistency check...")
    
    def calculate_distribution(df):
        total = len(df)
        dist = df['label'].value_counts().sort_index()
        return {label: count/total for label, count in dist.items()}
    
    train_dist = calculate_distribution(train_df)
    val_dist = calculate_distribution(val_df)
    test_dist = calculate_distribution(test_df)
    
    print("Class distribution in each set:")
    print(f"{'Category':<20} {'Train':<10} {'Val':<10} {'Test':<10}")
    print("-" * 50)
    
    for label in sorted(train_dist.keys()):
        print(f"{label:<20} {train_dist[label]:<10.3f} {val_dist[label]:<10.3f} {test_dist[label]:<10.3f}")
    
    # 计算分布差异
    max_diff = 0
    for label in train_dist.keys():
        diff = max(abs(train_dist[label] - val_dist[label]),
                  abs(train_dist[label] - test_dist[label]),
                  abs(val_dist[label] - test_dist[label]))
        max_diff = max(max_diff, diff)
    
    print(f"\nMaximum distribution difference: {max_diff:.3f}")
    if max_diff < 0.05:
        print("✅ Data distribution consistency is good")
    else:
        print("⚠️   Data distribution has significant differences")
    
    print(f"\nValidation report saved to: {processed_data_dir}/data_split_validation.png")
    print("✅ Data preprocessing validation completed!")
    
    return True

if __name__ == "__main__":
    validate_data_splits() 