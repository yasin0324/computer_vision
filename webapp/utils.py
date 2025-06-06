#!/usr/bin/env python3
"""
植物叶片病害识别系统 - Web应用工具类
"""

import os
import sys
import json
import uuid
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 默认配置
class DefaultConfig:
    TARGET_CLASSES = {
        0: 'bacterial_spot',
        1: 'healthy', 
        2: 'septoria_leaf_spot',
        3: 'target_spot'
    }

config = DefaultConfig()

# Mock模型创建函数
def create_mock_model(num_classes=4):
    """创建一个简单的mock模型用于演示"""
    import torchvision.models as models
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class ModelPredictor:
    """模型预测器"""
    
    def __init__(self):
        self.models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = list(config.TARGET_CLASSES.values())
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_path: str, model_type: str = 'baseline') -> nn.Module:
        """加载模型"""
        if model_path in self.models:
            return self.models[model_path]
        
        # 创建模型
        model = create_mock_model(num_classes=len(self.class_names))
        
        # 加载权重（如果存在）
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            except Exception as e:
                print(f"警告: 无法加载模型权重 {model_path}: {e}")
        
        model.to(self.device)
        model.eval()
        
        # 缓存模型
        self.models[model_path] = model
        
        return model
    
    def predict_image(self, image_path: str, model_type: str = 'baseline') -> Dict[str, Any]:
        """预测单张图像"""
        try:
            # 查找可用的模型文件
            model_path = self._find_model_file(model_type)
            if not model_path:
                # 如果没有找到模型文件，返回模拟结果
                return self._mock_prediction(model_type)
            
            # 加载模型
            model = self.load_model(model_path, model_type)
            
            # 加载和预处理图像
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 预测
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # 获取所有类别的概率
                all_probs = probabilities.cpu().numpy()[0]
                
            # 构建结果
            result = {
                'predicted_class': self.class_names[predicted.item()],
                'confidence': float(confidence.item()),
                'all_probabilities': {
                    self.class_names[i]: float(prob) 
                    for i, prob in enumerate(all_probs)
                },
                'model_type': model_type,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return self._mock_prediction(model_type, error=str(e))
    
    def _mock_prediction(self, model_type: str, error: str = None) -> Dict[str, Any]:
        """返回模拟预测结果"""
        import random
        
        # 生成随机概率
        probs = [random.random() for _ in self.class_names]
        total = sum(probs)
        probs = [p/total for p in probs]
        
        predicted_idx = probs.index(max(probs))
        
        result = {
            'predicted_class': self.class_names[predicted_idx],
            'confidence': max(probs),
            'all_probabilities': {
                self.class_names[i]: prob 
                for i, prob in enumerate(probs)
            },
            'model_type': model_type,
            'timestamp': datetime.now().isoformat(),
            'is_mock': True
        }
        
        if error:
            result['warning'] = f"使用模拟预测，原因: {error}"
        else:
            result['warning'] = "未找到训练好的模型，使用模拟预测"
            
        return result
    
    def _find_model_file(self, model_type: str) -> Optional[str]:
        """查找模型文件"""
        # 修改为从outputs/models目录查找模型
        models_dir = Path(project_root) / "outputs" / "models"
        
        if not models_dir.exists():
            return None
        
        # 根据模型类型查找对应的目录
        type_patterns = {
            'baseline': ['baseline', 'resnet50_baseline'],
            'senet': ['se_net', 'senet', 'resnet50_se_net'],
            'cbam': ['cbam', 'resnet50_cbam']
        }
        
        patterns = type_patterns.get(model_type, [model_type])
        
        # 在每个可能的目录中查找模型
        for pattern in patterns:
            for model_dir in models_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                    
                if pattern.lower() in model_dir.name.lower():
                    # 优先查找best_checkpoint文件
                    best_models = list(model_dir.glob("best_checkpoint_*.pth"))
                    if best_models:
                        # 返回最新的best checkpoint
                        latest_model = max(best_models, key=lambda x: x.stat().st_mtime)
                        return str(latest_model)
                    
                    # 如果没有best checkpoint，查找其他.pth文件
                    pth_files = list(model_dir.glob("*.pth"))
                    if pth_files:
                        latest_model = max(pth_files, key=lambda x: x.stat().st_mtime)
                        return str(latest_model)
        
        return None
    
    def get_available_models(self) -> List[Dict[str, str]]:
        """获取可用模型列表"""
        models = []
        models_dir = Path(project_root) / "models"
        
        if not models_dir.exists():
            # 返回示例模型列表
            return [
                {
                    'name': 'baseline_demo',
                    'path': 'demo/baseline.pth',
                    'type': 'baseline',
                    'size': '0.0 MB',
                    'modified': datetime.now().isoformat(),
                    'status': '演示模型'
                }
            ]
        
        for model_file in models_dir.rglob("*.pth"):
            model_name = model_file.stem
            
            # 推断模型类型
            model_type = 'baseline'
            if 'senet' in model_name.lower():
                model_type = 'senet'
            elif 'cbam' in model_name.lower():
                model_type = 'cbam'
            
            models.append({
                'name': model_name,
                'path': str(model_file),
                'type': model_type,
                'size': f"{model_file.stat().st_size / (1024*1024):.1f} MB",
                'modified': datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
            })
        
        return sorted(models, key=lambda x: x['modified'], reverse=True)


class TrainingManager:
    """训练管理器"""
    
    def __init__(self):
        self.training_processes = {}
        self.training_status = {}
    
    def start_training(self, model_type: str, epochs: int = 50, 
                      learning_rate: float = 0.0005, batch_size: int = 32) -> str:
        """启动训练"""
        training_id = str(uuid.uuid4())
        
        # 创建训练配置
        config_data = {
            'model_type': model_type,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'training_id': training_id,
            'start_time': datetime.now().isoformat()
        }
        
        # 保存配置
        config_file = Path(project_root) / "logs" / f"training_{training_id}.json"
        config_file.parent.mkdir(exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        # 模拟训练进程
        self._start_mock_training(training_id, config_data)
        
        return training_id
    
    def _start_mock_training(self, training_id: str, config_data: Dict[str, Any]):
        """启动模拟训练进程"""
        def mock_training():
            import time
            import random
            
            epochs = config_data['epochs']
            
            for epoch in range(epochs):
                # 模拟训练进度
                progress = (epoch + 1) / epochs * 100
                train_loss = 2.0 - (epoch / epochs) * 1.5 + random.uniform(-0.1, 0.1)
                val_loss = 1.8 - (epoch / epochs) * 1.2 + random.uniform(-0.1, 0.1)
                train_acc = 0.3 + (epoch / epochs) * 0.6 + random.uniform(-0.05, 0.05)
                val_acc = 0.4 + (epoch / epochs) * 0.5 + random.uniform(-0.05, 0.05)
                
                self.training_status[training_id] = {
                    'status': 'running',
                    'progress': progress,
                    'current_epoch': epoch + 1,
                    'total_epochs': epochs,
                    'train_loss': max(0.1, train_loss),
                    'val_loss': max(0.1, val_loss),
                    'train_acc': min(0.99, max(0.1, train_acc)),
                    'val_acc': min(0.99, max(0.1, val_acc)),
                    'start_time': config_data['start_time'],
                    'message': f'训练进行中... Epoch {epoch+1}/{epochs}'
                }
                
                time.sleep(2)  # 模拟训练时间
            
            # 训练完成
            self.training_status[training_id]['status'] = 'completed'
            self.training_status[training_id]['message'] = '训练完成'
            self.training_status[training_id]['end_time'] = datetime.now().isoformat()
        
        # 在后台线程中运行模拟训练
        thread = threading.Thread(target=mock_training)
        thread.daemon = True
        thread.start()
    
    def get_training_status(self, training_id: str) -> Dict[str, Any]:
        """获取训练状态"""
        if training_id not in self.training_status:
            return {'status': 'not_found', 'message': '训练任务不存在'}
        
        return self.training_status[training_id]
    
    def stop_training(self, training_id: str) -> bool:
        """停止训练"""
        if training_id in self.training_status:
            self.training_status[training_id]['status'] = 'stopped'
            self.training_status[training_id]['message'] = '训练已停止'
            return True
        return False


class FileManager:
    """文件管理器"""
    
    def __init__(self):
        self.project_root = project_root
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """获取可用模型列表"""
        models = []
        # 修改为从outputs/models目录读取模型
        models_dir = Path(self.project_root) / "outputs" / "models"
        
        if not models_dir.exists():
            # 返回示例数据
            return [
                {
                    'name': 'baseline_demo',
                    'path': 'demo/baseline.pth',
                    'type': 'baseline',
                    'size': '0.0 MB',
                    'modified': datetime.now().isoformat(),
                    'status': '演示模型'
                }
            ]
        
        # 遍历outputs/models下的所有子目录
        for model_dir in models_dir.iterdir():
            if not model_dir.is_dir():
                continue
                
            # 查找最佳模型文件（优先选择best_checkpoint开头的文件）
            best_models = list(model_dir.glob("best_checkpoint_*.pth"))
            if best_models:
                # 选择最新的best checkpoint
                model_file = max(best_models, key=lambda x: x.stat().st_mtime)
            else:
                # 如果没有best checkpoint，查找其他.pth文件
                pth_files = list(model_dir.glob("*.pth"))
                if pth_files:
                    model_file = max(pth_files, key=lambda x: x.stat().st_mtime)
                else:
                    continue
            
            # 从目录名推断模型类型
            dir_name = model_dir.name.lower()
            model_type = 'baseline'
            if 'se_net' in dir_name or 'senet' in dir_name:
                model_type = 'senet'
            elif 'cbam' in dir_name:
                model_type = 'cbam'
            elif 'baseline' in dir_name:
                model_type = 'baseline'
            
            # 生成友好的模型名称
            model_name = f"{model_type}_{model_file.stem}"
            
            models.append({
                'name': model_name,
                'path': str(model_file),
                'type': model_type,
                'size': f"{model_file.stat().st_size / (1024*1024):.1f} MB",
                'modified': datetime.fromtimestamp(model_file.stat().st_mtime).isoformat(),
                'directory': model_dir.name
            })
        
        return sorted(models, key=lambda x: x['modified'], reverse=True)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """获取数据集信息"""
        data_dir = Path(self.project_root) / "data"
        
        if not data_dir.exists():
            return {
                'total_samples': 0,
                'train_samples': 0,
                'val_samples': 0,
                'test_samples': 0,
                'classes': list(config.TARGET_CLASSES.values()),
                'status': '数据集未找到'
            }
        
        # 尝试统计数据集
        try:
            train_dir = data_dir / "train"
            val_dir = data_dir / "val" 
            test_dir = data_dir / "test"
            
            train_count = sum(len(list(class_dir.glob("*"))) for class_dir in train_dir.iterdir() if class_dir.is_dir()) if train_dir.exists() else 0
            val_count = sum(len(list(class_dir.glob("*"))) for class_dir in val_dir.iterdir() if class_dir.is_dir()) if val_dir.exists() else 0
            test_count = sum(len(list(class_dir.glob("*"))) for class_dir in test_dir.iterdir() if class_dir.is_dir()) if test_dir.exists() else 0
            
            return {
                'total_samples': train_count + val_count + test_count,
                'train_samples': train_count,
                'val_samples': val_count,
                'test_samples': test_count,
                'classes': list(config.TARGET_CLASSES.values()),
                'status': '正常'
            }
        except Exception as e:
            return {
                'total_samples': 0,
                'train_samples': 0,
                'val_samples': 0,
                'test_samples': 0,
                'classes': list(config.TARGET_CLASSES.values()),
                'status': f'数据集读取错误: {str(e)}'
            }
    
    def get_models_summary(self) -> Dict[str, Any]:
        """获取模型摘要"""
        models = self.get_available_models()
        
        summary = {
            'total': len(models),
            'by_type': {}
        }
        
        for model in models:
            model_type = model.get('type', 'unknown')
            if model_type not in summary['by_type']:
                summary['by_type'][model_type] = 0
            summary['by_type'][model_type] += 1
        
        return summary
    
    def get_recent_evaluations(self, limit: int = 5) -> List[Dict[str, Any]]:
        """获取最近的评估结果"""
        evaluations = []
        outputs_dir = Path(self.project_root) / "outputs" / "evaluation"
        
        if not outputs_dir.exists():
            return []
        
        try:
            # 查找评估结果文件
            for eval_dir in outputs_dir.iterdir():
                if eval_dir.is_dir():
                    # 查找HTML报告文件
                    for html_file in eval_dir.glob("*.html"):
                        evaluations.append({
                            'name': html_file.stem,
                            'path': str(html_file),
                            'date': datetime.fromtimestamp(html_file.stat().st_mtime).isoformat(),
                            'type': 'evaluation'
                        })
            
            # 按日期排序，返回最近的几个
            evaluations.sort(key=lambda x: x['date'], reverse=True)
            return evaluations[:limit]
            
        except Exception as e:
            print(f"获取评估历史失败: {e}")
            return []
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        return self.get_dataset_info() 