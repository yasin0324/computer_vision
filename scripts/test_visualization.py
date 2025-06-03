"""
测试可视化功能的简单脚本
"""

import matplotlib.pyplot as plt
import numpy as np

def test_chinese_display():
    """测试中文显示问题"""
    print("Testing visualization with English text...")
    
    # 创建测试数据
    categories = ['bacterial_spot', 'septoria_leaf_spot', 'target_spot', 'healthy']
    counts = [2127, 1771, 1404, 1591]
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    
    # 使用英文标题和标签
    plt.title('Tomato Spot Disease Dataset Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Disease Category', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # 添加数值标签
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('test_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualization test completed!")
    print("If you can see the chart with proper English text, the fix is working.")

if __name__ == "__main__":
    test_chinese_display() 