"""
番茄叶斑病细粒度识别实验配置文件
"""

import os
from pathlib import Path

class Config:
    """实验配置类"""
    
    # 数据相关配置
    DATA_ROOT = "data/PlantVillage"
    PROCESSED_DATA_DIR = "processed_data"
    
    # 目标类别（番茄叶斑病细粒度识别）
    TARGET_CLASSES = {
        'Tomato_Bacterial_spot': 'bacterial_spot',
        'Tomato_Septoria_leaf_spot': 'septoria_leaf_spot', 
        'Tomato__Target_Spot': 'target_spot',
        'Tomato_healthy': 'healthy'
    }
    
    # 数据划分比例
    TEST_SIZE = 0.2      # 测试集比例
    VAL_SIZE = 0.2       # 验证集比例
    RANDOM_STATE = 42    # 随机种子
    
    # 图像处理参数
    INPUT_SIZE = 224     # 输入图像尺寸
    BATCH_SIZE = 32      # 批次大小
    NUM_WORKERS = 4      # 数据加载线程数
    
    # 数据增强参数
    AUGMENTATION = {
        'horizontal_flip_prob': 0.5,
        'vertical_flip_prob': 0.3,
        'rotation_degrees': 15,
        'color_jitter': {
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1
        }
    }
    
    # 模型相关配置
    NUM_CLASSES = len(TARGET_CLASSES)
    PRETRAINED = True
    
    # 训练相关配置
    EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    STEP_SIZE = 30       # 学习率衰减步长
    GAMMA = 0.1          # 学习率衰减因子
    
    # 早停配置
    EARLY_STOPPING = {
        'patience': 15,
        'min_delta': 0.001
    }
    
    # 输出目录
    OUTPUT_DIR = "experiments"
    MODEL_SAVE_DIR = "models"
    LOGS_DIR = "logs"
    
    # 设备配置
    DEVICE = "cuda" if os.system("nvidia-smi") == 0 else "cpu"
    
    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        dirs = [
            cls.PROCESSED_DATA_DIR,
            cls.OUTPUT_DIR,
            cls.MODEL_SAVE_DIR,
            cls.LOGS_DIR
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(exist_ok=True)
            
    @classmethod
    def get_experiment_name(cls, model_name, attention_type=None):
        """生成实验名称"""
        if attention_type:
            return f"{model_name}_{attention_type}_tomato_spot"
        else:
            return f"{model_name}_baseline_tomato_spot"
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("=== Experiment Configuration ===")
        print(f"Data root directory: {cls.DATA_ROOT}")
        print(f"Number of target classes: {cls.NUM_CLASSES}")
        print(f"Image size: {cls.INPUT_SIZE}x{cls.INPUT_SIZE}")
        print(f"Batch size: {cls.BATCH_SIZE}")
        print(f"Training epochs: {cls.EPOCHS}")
        print(f"Learning rate: {cls.LEARNING_RATE}")
        print(f"Device: {cls.DEVICE}")
        print("=" * 20)


# 创建全局配置实例
config = Config() 