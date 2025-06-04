import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from collections import Counter
import shutil
from pathlib import Path
import random
from ..config import config

class TomatoSpotDiseaseDataset:
    """
    Tomato Spot Disease Fine-grained Recognition Data Preprocessing Class
    """
    
    def __init__(self, data_root="data/PlantVillage"):
        self.data_root = data_root
        self.target_classes = {
            'Tomato_Bacterial_spot': 'bacterial_spot',
            'Tomato_Septoria_leaf_spot': 'septoria_leaf_spot', 
            'Tomato__Target_Spot': 'target_spot',
            'Tomato_healthy': 'healthy'  # 添加健康对照组
        }
        
        # 数据统计
        self.class_counts = {}
        self.image_paths = []
        self.labels = []
        
        print("=== Tomato Spot Disease Fine-grained Recognition Data Preprocessing ===")
        print(f"Target classes: {list(self.target_classes.values())}")
        
    def collect_data(self):
        """收集目标类别的所有图像路径和标签"""
        print("\n1. Collecting data...")
        
        for class_dir, class_label in self.target_classes.items():
            class_path = os.path.join(self.data_root, class_dir)
            
            if not os.path.exists(class_path):
                print(f"Warning: Directory does not exist {class_path}")
                continue
                
            # 获取该类别的所有图像文件
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            self.class_counts[class_label] = len(image_files)
            
            # 添加到总列表
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                self.image_paths.append(img_path)
                self.labels.append(class_label)
        
        print(f"Total image count: {len(self.image_paths)}")
        print("Class counts:")
        for class_name, count in self.class_counts.items():
            print(f"  {class_name}: {count} images")
            
        return self.image_paths, self.labels
    
    def analyze_data_distribution(self):
        """分析数据分布"""
        print("\n2. Analyzing data distribution...")
        
        # 计算数据平衡性
        counts = list(self.class_counts.values())
        min_count, max_count = min(counts), max(counts)
        balance_ratio = min_count / max_count
        
        print(f"Data balance ratio: {balance_ratio:.3f}")
        if balance_ratio < 0.5:
            print("⚠️  数据不平衡，建议进行数据增强或重采样")
        else:
            print("✅ 数据相对平衡")
            
        # 可视化数据分布
        plt.figure(figsize=(10, 6))
        classes = list(self.class_counts.keys())
        counts = list(self.class_counts.values())
        
        bars = plt.bar(classes, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        plt.title('Tomato Spot Disease Dataset Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Disease Category', fontsize=12)
        plt.ylabel('Number of Images', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('tomato_spot_disease_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return balance_ratio
    
    def check_image_quality(self, sample_size=100):
        """检查图像质量"""
        print(f"\n3. Image quality check (sample size: {sample_size})...")
        
        # 随机采样检查
        sample_indices = random.sample(range(len(self.image_paths)), 
                                     min(sample_size, len(self.image_paths)))
        
        corrupted_images = []
        image_stats = {
            'widths': [], 'heights': [], 'channels': [], 
            'file_sizes': [], 'sharpness': []
        }
        
        for idx in sample_indices:
            img_path = self.image_paths[idx]
            try:
                # 使用PIL检查基本信息
                with Image.open(img_path) as img:
                    width, height = img.size
                    channels = len(img.getbands())
                    file_size = os.path.getsize(img_path)
                    
                    # 使用OpenCV计算清晰度
                    cv_img = cv2.imread(img_path)
                    if cv_img is not None:
                        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                        
                        image_stats['widths'].append(width)
                        image_stats['heights'].append(height)
                        image_stats['channels'].append(channels)
                        image_stats['file_sizes'].append(file_size)
                        image_stats['sharpness'].append(sharpness)
                    else:
                        corrupted_images.append(img_path)
                        
            except Exception as e:
                corrupted_images.append(img_path)
                print(f"Corrupted image: {os.path.basename(img_path)} - {str(e)}")
        
        # 统计结果
        if image_stats['widths']:
            print(f"Image dimensions: {min(image_stats['widths'])}x{min(image_stats['heights'])} ~ "
                  f"{max(image_stats['widths'])}x{max(image_stats['heights'])}")
            print(f"Average dimensions: {np.mean(image_stats['widths']):.0f}x{np.mean(image_stats['heights']):.0f}")
            print(f"Channel count: {min(image_stats['channels'])} ~ {max(image_stats['channels'])}")
            print(f"File size: {min(image_stats['file_sizes'])/1024:.1f}KB ~ {max(image_stats['file_sizes'])/1024:.1f}KB")
            print(f"Sharpness range: {min(image_stats['sharpness']):.1f} ~ {max(image_stats['sharpness']):.1f}")
            print(f"Corrupted image count: {len(corrupted_images)}")
            
        return image_stats, corrupted_images
    
    def create_data_splits(self, test_size=0.2, val_size=0.2, random_state=42):
        """创建训练/验证/测试数据划分"""
        print(f"\n4. Data splitting (train:{1-test_size-val_size:.0%}, val:{val_size:.0%}, test:{test_size:.0%})...")
        
        # 转换为DataFrame便于处理
        df = pd.DataFrame({
            'image_path': self.image_paths,
            'label': self.labels
        })
        
        # 分层划分：先分出测试集
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, 
            stratify=df['label'], 
            random_state=random_state
        )
        
        # 再从训练+验证集中分出验证集
        val_size_adjusted = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size_adjusted,
            stratify=train_val_df['label'],
            random_state=random_state
        )
        
        # 统计各集合的类别分布
        splits = {'train': train_df, 'val': val_df, 'test': test_df}
        
        print("Class distribution in each dataset:")
        for split_name, split_df in splits.items():
            print(f"\n{split_name.upper()} ({len(split_df)} images):")
            class_dist = split_df['label'].value_counts().sort_index()
            for class_name, count in class_dist.items():
                percentage = count / len(split_df) * 100
                print(f"  {class_name}: {count} images ({percentage:.1f}%)")
        
        return train_df, val_df, test_df
    
    def save_data_splits(self, train_df, val_df, test_df, output_dir="processed_data"):
        """保存数据划分结果"""
        print(f"\n5. Saving data splitting to {output_dir}/...")
        
        # 创建输出目录
        Path(output_dir).mkdir(exist_ok=True)
        
        # 保存CSV文件
        train_df.to_csv(f"{output_dir}/train_split.csv", index=False)
        val_df.to_csv(f"{output_dir}/val_split.csv", index=False)
        test_df.to_csv(f"{output_dir}/test_split.csv", index=False)
        
        # 保存类别映射
        label_to_idx = {label: idx for idx, label in enumerate(sorted(self.target_classes.values()))}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        
        import json
        with open(f"{output_dir}/class_mapping.json", 'w') as f:
            json.dump({
                'label_to_idx': label_to_idx,
                'idx_to_label': idx_to_label,
                'class_names': list(self.target_classes.values())
            }, f, indent=2)
        
        print("Saving completed:")
        print(f"  - train_split.csv: {len(train_df)} images")
        print(f"  - val_split.csv: {len(val_df)} images") 
        print(f"  - test_split.csv: {len(test_df)} images")
        print(f"  - class_mapping.json: Class mapping file")
        
        return output_dir


class TomatoSpotDataset(Dataset):
    """
    PyTorch Dataset类，用于加载番茄叶斑病数据
    """
    
    def __init__(self, dataframe, transform=None, label_to_idx=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        
        # 如果没有提供标签映射，自动创建
        if label_to_idx is None:
            unique_labels = sorted(dataframe['label'].unique())
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_to_idx = label_to_idx
            
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = row['image_path']
        label_str = row['label']
        
        # 加载图像
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回一个黑色图像作为fallback
            image = Image.new('RGB', (256, 256), (0, 0, 0))
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
            
        # 转换标签为数字
        label = self.label_to_idx[label_str]
        
        return image, label, image_path


def get_data_transforms(input_size=224, augment=True):
    """
    获取数据变换管道
    """
    # 基础变换（用于验证和测试）
    base_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet标准化
    ])
    
    if not augment:
        return base_transform
    
    # 训练时的数据增强
    train_transform = transforms.Compose([
        transforms.Resize((input_size + 32, input_size + 32)),  # 稍大一些用于随机裁剪
        transforms.RandomCrop((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform


def create_data_loaders(train_df, val_df, test_df, label_to_idx, 
                       batch_size=32, input_size=224, num_workers=4):
    """
    创建数据加载器
    """
    print(f"\n6. Creating data loaders (batch_size={batch_size})...")
    
    # 获取变换
    train_transform = get_data_transforms(input_size, augment=True)
    val_test_transform = get_data_transforms(input_size, augment=False)
    
    # 创建数据集
    train_dataset = TomatoSpotDataset(train_df, train_transform, label_to_idx)
    val_dataset = TomatoSpotDataset(val_df, val_test_transform, label_to_idx)
    test_dataset = TomatoSpotDataset(test_df, val_test_transform, label_to_idx)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"Train set: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"Validation set: {len(val_dataset)} images, {len(val_loader)} batches")
    print(f"Test set: {len(test_dataset)} images, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader, train_dataset.idx_to_label


def visualize_samples(data_loader, idx_to_label, num_samples=16, save_path="sample_images.png"):
    """
    可视化数据样本
    """
    print(f"\n7. Visualizing data samples...")
    
    # 获取一批数据
    data_iter = iter(data_loader)
    images, labels, paths = next(data_iter)
    
    # 反标准化用于显示
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        # 反标准化
        img = images[i] * std + mean
        img = torch.clamp(img, 0, 1)
        
        # 转换为numpy并调整维度
        img_np = img.permute(1, 2, 0).numpy()
        
        axes[i].imshow(img_np)
        axes[i].set_title(f"{idx_to_label[labels[i].item()]}", fontsize=10)
        axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Tomato Spot Disease Data Samples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Sample images saved to: {save_path}")


def main():
    """
    主函数：执行完整的数据预处理流程
    """
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 1. 初始化数据集处理器，使用配置文件中的数据路径
    dataset_processor = TomatoSpotDiseaseDataset(data_root=config.DATA_ROOT)
    
    # 2. 收集数据
    image_paths, labels = dataset_processor.collect_data()
    
    # 检查是否收集到数据
    if len(image_paths) == 0:
        print("❌ No images found! Please check:")
        print(f"   1. Data directory exists: {config.DATA_ROOT}")
        print("   2. Target class directories exist:")
        for class_dir in dataset_processor.target_classes.keys():
            class_path = os.path.join(config.DATA_ROOT, class_dir)
            print(f"      - {class_path}")
        raise ValueError("No images found in the specified directories")
    
    # 3. 分析数据分布
    balance_ratio = dataset_processor.analyze_data_distribution()
    
    # 4. 检查图像质量
    image_stats, corrupted_images = dataset_processor.check_image_quality()
    
    # 5. 创建数据划分
    train_df, val_df, test_df = dataset_processor.create_data_splits(
        test_size=config.TEST_SIZE,
        val_size=config.VAL_SIZE,
        random_state=config.RANDOM_STATE
    )
    
    # 6. 保存数据划分
    output_dir = dataset_processor.save_data_splits(
        train_df, val_df, test_df, 
        output_dir=config.PROCESSED_DATA_DIR
    )
    
    # 7. 创建数据加载器
    import json
    with open(f"{output_dir}/class_mapping.json", 'r') as f:
        class_mapping = json.load(f)
    
    label_to_idx = class_mapping['label_to_idx']
    
    train_loader, val_loader, test_loader, idx_to_label = create_data_loaders(
        train_df, val_df, test_df, label_to_idx, 
        batch_size=config.BATCH_SIZE, 
        input_size=config.INPUT_SIZE,
        num_workers=config.NUM_WORKERS
    )
    
    # 8. 可视化样本
    visualize_samples(train_loader, idx_to_label, save_path=f"{output_dir}/sample_images.png")
    
    print("\n=== Data preprocessing completed ===")
    print(f"Processed data saved to: {output_dir}/")
    print("Next step: Start model training!")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader, 
        'test_loader': test_loader,
        'idx_to_label': idx_to_label,
        'label_to_idx': label_to_idx,
        'output_dir': output_dir
    }


if __name__ == "__main__":
    results = main() 