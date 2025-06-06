#!/usr/bin/env python3
"""
训练监控脚本
检查训练进度和状态
"""

import sys
import time
from pathlib import Path
import json
import pandas as pd

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def check_training_progress():
    """检查训练进度"""
    print("="*60)
    print("训练进度监控")
    print("="*60)
    
    # 检查模型目录
    models_dir = Path("outputs/models")
    if models_dir.exists():
        print(f"\n模型目录: {models_dir}")
        for exp_dir in models_dir.iterdir():
            if exp_dir.is_dir():
                print(f"  实验: {exp_dir.name}")
                
                # 检查检查点文件
                checkpoints = list(exp_dir.glob("*.pth"))
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
                    print(f"    最新检查点: {latest_checkpoint.name}")
                    print(f"    修改时间: {time.ctime(latest_checkpoint.stat().st_mtime)}")
                
                # 检查训练历史
                history_file = exp_dir / "training_history.json"
                if history_file.exists():
                    try:
                        with open(history_file, 'r') as f:
                            history = json.load(f)
                        
                        if history.get('val_acc'):
                            epochs_completed = len(history['val_acc'])
                            best_val_acc = max(history['val_acc'])
                            latest_val_acc = history['val_acc'][-1]
                            
                            print(f"    已完成轮数: {epochs_completed}")
                            print(f"    最佳验证准确率: {best_val_acc:.4f}")
                            print(f"    最新验证准确率: {latest_val_acc:.4f}")
                    except Exception as e:
                        print(f"    无法读取训练历史: {e}")
    else:
        print("未找到模型目录")
    
    # 检查日志目录
    logs_dir = Path("outputs/logs")
    if logs_dir.exists():
        print(f"\n日志目录: {logs_dir}")
        for exp_dir in logs_dir.iterdir():
            if exp_dir.is_dir():
                print(f"  实验: {exp_dir.name}")
                
                # 检查最新日志文件
                log_files = list(exp_dir.glob("*.log"))
                if log_files:
                    latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                    print(f"    最新日志: {latest_log.name}")
                    print(f"    修改时间: {time.ctime(latest_log.stat().st_mtime)}")
                    
                    # 读取最后几行日志
                    try:
                        with open(latest_log, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                        
                        if lines:
                            print("    最新日志内容:")
                            for line in lines[-5:]:  # 显示最后5行
                                print(f"      {line.strip()}")
                    except Exception as e:
                        print(f"    无法读取日志文件: {e}")
    else:
        print("未找到日志目录")


def main():
    """主函数"""
    print("开始监控训练进度...")
    check_training_progress()
    print(f"\n✅ 监控完成!")


if __name__ == "__main__":
    main() 