#!/usr/bin/env python3
import sys
import os
import json
import torch
from pathlib import Path
from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import pandas as pd

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
os.chdir(str(project_root))

def evaluate_model(model_path, model_type, model_name):
    try:
        from src.data.dataset import TomatoSpotDataset
        from src.models.baseline import create_resnet_baseline
        from src.models.attention_models import create_senet_model, create_cbam_model
        from src.config.config import config
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_csv_path = project_root / "data" / "processed" / "test_split.csv"
        test_df = pd.read_csv(test_csv_path)
        
        label_to_idx = {label: idx for idx, label in enumerate(config.TARGET_CLASSES.values())}
        
        test_dataset = TomatoSpotDataset(
            dataframe=test_df,
            transform=test_transform,
            label_to_idx=label_to_idx
        )
        
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        num_classes = len(config.TARGET_CLASSES)
        if model_type == "baseline":
            model = create_resnet_baseline(num_classes=num_classes)
        elif model_type == "senet":
            model = create_senet_model(num_classes=num_classes)
        elif model_type == "cbam":
            model = create_cbam_model(num_classes=num_classes)
        else:
            raise ValueError(f"未知模型类型: {model_type}")
        
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        all_preds = []
        all_labels = []
        total_time = 0
        
        with torch.no_grad():
            for images, labels, _ in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                start_time = time.time()
                outputs = model(images)
                end_time = time.time()
                
                total_time += (end_time - start_time)
                
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        class_names = list(config.TARGET_CLASSES.values())
        
        report = classification_report(all_labels, all_preds, 
                                     target_names=class_names, 
                                     output_dict=True)
        
        avg_inference_time = (total_time / len(test_loader.dataset)) * 1000
        file_size_mb = Path(model_path).stat().st_size / (1024*1024)
        
        result = {
            'model_name': model_name,
            'model_type': model_type,
            'model_path': model_path,
            'accuracy': float(accuracy),
            'f1_score': float(report['macro avg']['f1-score']),
            'precision': float(report['macro avg']['precision']),
            'recall': float(report['macro avg']['recall']),
            'file_size_mb': round(file_size_mb, 1),
            'inference_time_ms': round(avg_inference_time, 2),
            'evaluation_date': datetime.now().isoformat(),
            'status': '真实评估结果'
        }
        
        return result
        
    except Exception as e:
        return {
            'model_name': model_name,
            'model_type': model_type,
            'model_path': model_path,
            'error': str(e),
            'status': f'评估失败: {str(e)}'
        }

if __name__ == "__main__":
    try:
        if len(sys.argv) != 4:
            result = {'error': '参数错误', 'status': '参数错误'}
            print(json.dumps(result))
            sys.exit(1)
        
        model_path = sys.argv[1]
        model_type = sys.argv[2]
        model_name = sys.argv[3]
        
        if not Path(model_path).exists():
            result = {
                'model_name': model_name,
                'error': f"文件不存在: {model_path}",
                'status': '文件不存在'
            }
            print(json.dumps(result))
            sys.exit(1)
        
        result = evaluate_model(model_path, model_type, model_name)
        print(json.dumps(result))
        
    except Exception as e:
        result = {'error': str(e), 'status': '脚本异常'}
        print(json.dumps(result))
        sys.exit(1) 