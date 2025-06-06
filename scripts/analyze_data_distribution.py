#!/usr/bin/env python3
"""
数据分布分析脚本
分析训练、验证、测试集的数据分布，检查潜在问题
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def load_data_splits():
    """加载所有数据划分"""
    data_dir = Path(project_root) / "data" / "processed"
    
    train_df = pd.read_csv(data_dir / "train_split.csv")
    val_df = pd.read_csv(data_dir / "val_split.csv")
    test_df = pd.read_csv(data_dir / "test_split.csv")
    
    return train_df, val_df, test_df


def analyze_class_distribution(train_df, val_df, test_df):
    """分析类别分布"""
    print("="*60)
    print("类别分布分析")
    print("="*60)
    
    # 计算各集合的类别分布
    train_dist = train_df['label'].value_counts().sort_index()
    val_dist = val_df['label'].value_counts().sort_index()
    test_dist = test_df['label'].value_counts().sort_index()
    
    # 创建分布表
    distribution_df = pd.DataFrame({
        'Train_Count': train_dist,
        'Val_Count': val_dist,
        'Test_Count': test_dist
    })
    
    # 计算百分比
    distribution_df['Train_Pct'] = (distribution_df['Train_Count'] / len(train_df) * 100).round(2)
    distribution_df['Val_Pct'] = (distribution_df['Val_Count'] / len(val_df) * 100).round(2)
    distribution_df['Test_Pct'] = (distribution_df['Test_Count'] / len(test_df) * 100).round(2)
    
    print("\n类别分布统计:")
    print(distribution_df)
    
    # 检查分布差异
    print("\n分布差异分析:")
    for class_name in distribution_df.index:
        train_pct = distribution_df.loc[class_name, 'Train_Pct']
        val_pct = distribution_df.loc[class_name, 'Val_Pct']
        test_pct = distribution_df.loc[class_name, 'Test_Pct']
        
        val_diff = abs(train_pct - val_pct)
        test_diff = abs(train_pct - test_pct)
        
        print(f"{class_name}:")
        print(f"  训练集: {train_pct:.2f}%, 验证集: {val_pct:.2f}% (差异: {val_diff:.2f}%)")
        print(f"  训练集: {train_pct:.2f}%, 测试集: {test_pct:.2f}% (差异: {test_diff:.2f}%)")
        
        if val_diff > 5.0 or test_diff > 5.0:
            print(f"  ⚠️  警告: {class_name} 类别分布差异较大!")
    
    return distribution_df


def check_data_leakage(train_df, val_df, test_df):
    """检查数据泄露"""
    print("\n" + "="*60)
    print("数据泄露检查")
    print("="*60)
    
    # 提取文件名（去除路径）
    train_files = set(Path(p).name for p in train_df['image_path'])
    val_files = set(Path(p).name for p in val_df['image_path'])
    test_files = set(Path(p).name for p in test_df['image_path'])
    
    # 检查重叠
    train_val_overlap = train_files & val_files
    train_test_overlap = train_files & test_files
    val_test_overlap = val_files & test_files
    
    print(f"训练集样本数: {len(train_files)}")
    print(f"验证集样本数: {len(val_files)}")
    print(f"测试集样本数: {len(test_files)}")
    
    print(f"\n重叠检查:")
    print(f"训练集-验证集重叠: {len(train_val_overlap)} 个文件")
    print(f"训练集-测试集重叠: {len(train_test_overlap)} 个文件")
    print(f"验证集-测试集重叠: {len(val_test_overlap)} 个文件")
    
    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("⚠️  发现数据泄露!")
        if train_val_overlap:
            print(f"训练集-验证集重叠文件示例: {list(train_val_overlap)[:5]}")
        if train_test_overlap:
            print(f"训练集-测试集重叠文件示例: {list(train_test_overlap)[:5]}")
        if val_test_overlap:
            print(f"验证集-测试集重叠文件示例: {list(val_test_overlap)[:5]}")
    else:
        print("✅ 未发现数据泄露")


def analyze_source_distribution(train_df, val_df, test_df):
    """分析数据来源分布"""
    print("\n" + "="*60)
    print("数据来源分析")
    print("="*60)
    
    def extract_source_info(df):
        """提取数据来源信息"""
        df = df.copy()
        df['filename'] = df['image_path'].apply(lambda x: Path(x).name)
        
        # 提取来源标识（文件名中的特殊标识）
        df['source'] = df['filename'].str.extract(r'___([A-Za-z.]+)_')
        df['source'] = df['source'].fillna('Unknown')
        
        return df
    
    train_df_ext = extract_source_info(train_df)
    val_df_ext = extract_source_info(val_df)
    test_df_ext = extract_source_info(test_df)
    
    # 分析每个类别的来源分布
    for label in train_df['label'].unique():
        print(f"\n{label} 类别的来源分布:")
        
        train_sources = train_df_ext[train_df_ext['label'] == label]['source'].value_counts()
        val_sources = val_df_ext[val_df_ext['label'] == label]['source'].value_counts()
        test_sources = test_df_ext[test_df_ext['label'] == label]['source'].value_counts()
        
        print(f"  训练集: {dict(train_sources)}")
        print(f"  验证集: {dict(val_sources)}")
        print(f"  测试集: {dict(test_sources)}")


def plot_distribution_comparison(distribution_df):
    """绘制分布比较图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制数量分布
    x = np.arange(len(distribution_df.index))
    width = 0.25
    
    ax1.bar(x - width, distribution_df['Train_Count'], width, label='训练集', alpha=0.8)
    ax1.bar(x, distribution_df['Val_Count'], width, label='验证集', alpha=0.8)
    ax1.bar(x + width, distribution_df['Test_Count'], width, label='测试集', alpha=0.8)
    
    ax1.set_xlabel('类别')
    ax1.set_ylabel('样本数量')
    ax1.set_title('各集合类别数量分布')
    ax1.set_xticks(x)
    ax1.set_xticklabels(distribution_df.index, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制百分比分布
    ax2.bar(x - width, distribution_df['Train_Pct'], width, label='训练集', alpha=0.8)
    ax2.bar(x, distribution_df['Val_Pct'], width, label='验证集', alpha=0.8)
    ax2.bar(x + width, distribution_df['Test_Pct'], width, label='测试集', alpha=0.8)
    
    ax2.set_xlabel('类别')
    ax2.set_ylabel('百分比 (%)')
    ax2.set_title('各集合类别百分比分布')
    ax2.set_xticks(x)
    ax2.set_xticklabels(distribution_df.index, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = Path(project_root) / "outputs" / "data_distribution_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n分布比较图已保存到: {output_path}")


def main():
    """主函数"""
    print("开始数据分布分析...")
    
    # 加载数据
    train_df, val_df, test_df = load_data_splits()
    
    print(f"\n数据集大小:")
    print(f"训练集: {len(train_df)} 样本")
    print(f"验证集: {len(val_df)} 样本")
    print(f"测试集: {len(test_df)} 样本")
    print(f"总计: {len(train_df) + len(val_df) + len(test_df)} 样本")
    
    # 分析类别分布
    distribution_df = analyze_class_distribution(train_df, val_df, test_df)
    
    # 检查数据泄露
    check_data_leakage(train_df, val_df, test_df)
    
    # 分析数据来源
    analyze_source_distribution(train_df, val_df, test_df)
    
    # 绘制分布图
    plot_distribution_comparison(distribution_df)
    
    # 保存分析结果
    output_dir = Path(project_root) / "outputs"
    distribution_df.to_csv(output_dir / "data_distribution_analysis.csv")
    
    print(f"\n✅ 数据分布分析完成!")
    print(f"详细结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()