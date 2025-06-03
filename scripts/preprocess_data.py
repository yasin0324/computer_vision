"""
运行数据预处理的脚本
"""

import sys
import os
from src.config import config
from src.data.preprocessing import main as preprocess_main

def main():
    """
    执行数据预处理
    """
    print("Starting tomato spot disease data preprocessing...")
    
    # 创建必要目录
    config.create_directories()
    
    # 打印配置
    config.print_config()
    
    # 检查数据目录是否存在
    if not os.path.exists(config.DATA_ROOT):
        print(f"Error: Data directory does not exist {config.DATA_ROOT}")
        print("Please ensure PlantVillage dataset is correctly downloaded to data/ directory")
        sys.exit(1)
    
    try:
        # 执行预处理
        results = preprocess_main()
        
        print("\n✅ Data preprocessing completed successfully!")
        print(f"Results saved to: {results['output_dir']}")
        
        # 保存一些关键信息
        import json
        summary = {
            'num_classes': config.NUM_CLASSES,
            'class_names': list(config.TARGET_CLASSES.values()),
            'train_samples': len(results['train_loader'].dataset),
            'val_samples': len(results['val_loader'].dataset),
            'test_samples': len(results['test_loader'].dataset),
            'input_size': config.INPUT_SIZE,
            'batch_size': config.BATCH_SIZE
        }
        
        with open(f"{results['output_dir']}/preprocessing_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"Preprocessing summary saved to: {results['output_dir']}/preprocessing_summary.json")
        
    except Exception as e:
        print(f"❌ Data preprocessing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 