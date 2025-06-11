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
import logging
import cv2
import torch.nn.functional as F

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# --- 正确导入模型创建函数 ---
from src.models.baseline import create_resnet_baseline
from src.models.attention_models import ResNetSE, ResNetCBAM
from src.config.config import config

# 默认配置 (保留作为后备)
class DefaultConfig:
    TARGET_CLASSES = {
        0: 'bacterial_spot',
        1: 'healthy', 
        2: 'septoria_leaf_spot',
        3: 'target_spot'
    }
    INPUT_SIZE = 224
    NUM_CLASSES = 4

try:
    # 尝试从主配置文件导入
    from src.config.config import config
except ImportError:
    # 如果失败，使用这里的默认配置
    config = DefaultConfig()

class GradCAM:
    """Grad-CAM可视化器"""
    
    def __init__(self, model, target_layer, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向和反向钩子"""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, class_idx=None):
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_()
        
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward(retain_graph=True)
        
        if self.gradients is not None:
            weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
            cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
            cam = F.relu(cam)
            cam = cam - cam.min()
            cam = cam / cam.max() if cam.max() > 0 else cam
            return cam.detach().cpu().numpy(), class_idx
        else:
            return np.zeros((7, 7)), class_idx

class ModelPredictor:
    """模型预测器"""
    
    def __init__(self):
        self.models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = list(config.TARGET_CLASSES.values())
        logging.info(f"ModelPredictor initialized on device: {self.device}")
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_path: str, model_type: str = 'baseline') -> nn.Module:
        """加载模型"""
        if model_path in self.models:
            logging.info(f"DEBUG: Using cached model for: {model_path}")
            return self.models[model_path]
        
        # --- 根据模型类型正确创建模型 ---
        logging.info(f"DEBUG: Creating model architecture for type: '{model_type}'")
        if model_type == 'baseline':
            model = create_resnet_baseline(num_classes=config.NUM_CLASSES, pretrained=False)
        elif model_type == 'senet':
            model = ResNetSE(num_classes=config.NUM_CLASSES)
        elif model_type == 'cbam':
            model = ResNetCBAM(num_classes=config.NUM_CLASSES)
        else:
            logging.error(f"Unsupported model type: {model_type}. Falling back to baseline.")
            model = create_resnet_baseline(num_classes=config.NUM_CLASSES, pretrained=False)

        # 加载权重（如果存在）
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                state_dict = checkpoint.get('model_state_dict', checkpoint)

                # --- 针对不同模型结构，使用正确的加载方式 ---
                if model_type in ['senet', 'cbam']:
                    # 注意力模型权重需要加载到内部的 .model 属性
                    model.model.load_state_dict(state_dict)
                    logging.info(f"DEBUG: Successfully loaded weights into 'model.model' for {model_type}.")
                else:
                    # 基线模型直接加载
                    model.load_state_dict(state_dict)
                    logging.info("DEBUG: Successfully loaded weights into baseline model.")

            except Exception as e:
                logging.error(f"DEBUG: Failed to load model weights from {model_path}: {e}", exc_info=True)
        else:
            logging.warning(f"DEBUG: Model path does not exist: {model_path}. Using an untrained model.")

        model.to(self.device)
        model.eval()
        
        # 缓存模型
        self.models[model_path] = model
        
        return model
    
    def predict_image(self, image_path: str, model_type: str = 'baseline') -> Dict[str, Any]:
        """预测单张图像"""
        logging.info("--- [DEBUG] Starting Prediction ---")
        logging.info(f"DEBUG: Attempting to predict for model_type: '{model_type}'")
        try:
            # 查找可用的模型文件
            model_path = self._find_model_file(model_type)
            if not model_path:
                # 如果没有找到模型文件，返回模拟结果
                logging.warning(f"DEBUG: No model file found for type '{model_type}'. Returning mock prediction.")
                return self._mock_prediction(model_type)
            
            logging.info(f"DEBUG: Found model file to use: {model_path}")
            
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
                
            predicted_class = self.class_names[predicted.item()]
            confidence_val = float(confidence.item())
            logging.info(f"DEBUG: Prediction result: Class='{predicted_class}', Confidence={confidence_val:.4f}")
                
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
            
            logging.info("--- [DEBUG] Prediction Finished ---")
            # 返回结果和模型对象，方便后续使用
            return result, model
            
        except Exception as e:
            logging.error(f"DEBUG: An error occurred during prediction: {e}", exc_info=True)
            return self._mock_prediction(model_type, error=str(e))
    
    def generate_gradcam(self, model, image_path, pred_class_name, model_type: str):
        """为单张图像生成Grad-CAM"""
        try:
            # --- 根据模型类型确定目标层 ---
            target_layer = None
            if model_type == 'baseline':
                # 基线模型将ResNet层包装在 `backbone` 的 nn.Sequential 中
                # layer4 是其中的第8个元素（索引为7）
                if hasattr(model, 'backbone') and isinstance(model.backbone, nn.Sequential) and len(model.backbone) > 7:
                    target_layer = model.backbone[7]
                    logging.info("Found target layer for baseline model: model.backbone[7]")
            elif model_type in ['senet', 'cbam']:
                # 注意力模型的目标层在 model.layer4
                if hasattr(model, 'model') and hasattr(model.model, 'layer4'):
                    target_layer = model.model.layer4
                    logging.info(f"Found target layer for {model_type} model: model.model.layer4")

            if not target_layer:
                logging.warning(f"Grad-CAM: Could not find target layer for model type '{model_type}'.")
                return None

            grad_cam = GradCAM(model, target_layer, self.device)
            
            # 图像预处理
            img = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(img).unsqueeze(0)

            # 生成CAM
            class_idx = self.class_names.index(pred_class_name)
            cam, _ = grad_cam.generate_cam(input_tensor, class_idx)
            
            # 可视化
            img_resized = img.resize((224, 224))
            img_array = np.array(img_resized)
            
            cam_resized = cv2.resize(cam, (224, 224))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            superimposed = heatmap_rgb * 0.5 + img_array * 0.5
            superimposed = superimposed.astype(np.uint8)
            
            # 保存图像
            gradcam_dir = Path(project_root) / 'webapp' / 'static' / 'gradcam_results'
            gradcam_dir.mkdir(exist_ok=True)
            
            unique_filename = f"{uuid.uuid4()}.png"
            save_path = gradcam_dir / unique_filename
            
            Image.fromarray(superimposed).save(save_path)
            
            # 返回Web可访问的路径
            return f"/static/gradcam_results/{unique_filename}"

        except Exception as e:
            logging.error(f"Error generating Grad-CAM: {e}", exc_info=True)
            return None
    
    def _mock_prediction(self, model_type: str, error: str = None) -> Dict[str, Any]:
        """返回一个固定的、明确的模拟预测结果，用于清晰地指示错误"""
        
        # 固定返回第一个类别，低置信度，以明确表示是模拟结果
        predicted_class = self.class_names[0]
        confidence = 0.1 
        
        all_probabilities = {name: 0.0 for name in self.class_names}
        all_probabilities[predicted_class] = confidence

        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': all_probabilities,
            'model_type': model_type,
            'timestamp': datetime.now().isoformat(),
            'is_mock': True,
            'error': True # 明确标记这是一个错误/模拟结果
        }
        
        if error:
            result['warning'] = f"模型加载失败，返回模拟预测。原因: {error}"
        else:
            result['warning'] = "未找到或无法加载模型，返回模拟预测。"
            
        return result, None # Also return None for model
    
    def _find_model_file(self, model_type: str) -> Optional[str]:
        """查找模型文件"""
        models_dir = Path(project_root) / "outputs" / "models"
        logging.info(f"DEBUG: Searching for model type '{model_type}' in: {models_dir}")

        if not models_dir.exists():
            logging.warning(f"DEBUG: Models directory not found: {models_dir}")
            return None
        
        type_patterns = {
            'baseline': ['baseline', 'resnet50_baseline'],
            'senet': ['se_net', 'senet', 'resnet50_se_net'],
            'cbam': ['cbam', 'resnet50_cbam']
        }
        
        patterns = type_patterns.get(model_type, [model_type])
        logging.info(f"DEBUG: Using search patterns: {patterns}")
        
        for pattern in patterns:
            for model_dir in models_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                
                if pattern.lower() in model_dir.name.lower():
                    logging.info(f"DEBUG: Found matching directory for pattern '{pattern}': {model_dir.name}")
                    best_models = list(model_dir.glob("best_checkpoint_*.pth"))
                    if best_models:
                        latest_model = max(best_models, key=lambda x: x.stat().st_mtime)
                        logging.info(f"DEBUG: Found best checkpoint in directory: {latest_model}")
                        return str(latest_model)
                    
                    pth_files = list(model_dir.glob("*.pth"))
                    if pth_files:
                        latest_model = max(pth_files, key=lambda x: x.stat().st_mtime)
                        logging.info(f"DEBUG: Found latest .pth file in directory: {latest_model}")
                        return str(latest_model)
        
        logging.warning(f"DEBUG: No model file found for type '{model_type}' with patterns {patterns}")
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
        
        # 添加日志记录器
        self.logger = logging.getLogger("TrainingManager")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
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
        
        try:
            # 尝试启动实际训练进程
            self._start_real_training(training_id, config_data)
            
            # 检查训练是否成功启动
            if training_id in self.training_processes:
                return training_id
            
            # 如果训练进程没有成功启动，尝试使用模拟训练
            self._start_mock_training(training_id, config_data)
            self.training_status[training_id]['message'] += " (使用模拟训练，实际训练脚本启动失败)"
            
        except Exception as e:
            # 出现异常时，使用模拟训练
            self.logger.error(f"实际训练启动失败，使用模拟训练: {str(e)}")
            self._start_mock_training(training_id, config_data)
            self.training_status[training_id]['message'] = f"使用模拟训练 (实际训练脚本启动失败: {str(e)})"
        
        return training_id
    
    def _start_real_training(self, training_id: str, config_data: Dict[str, Any]):
        """启动实际训练进程"""
        model_type = config_data['model_type']
        
        # 根据模型类型选择对应的训练脚本
        script_map = {
            'baseline': 'train_baseline.py',
            'senet': 'train_se_net.py',
            'cbam': 'train_cbam.py'
        }
        
        script_name = script_map.get(model_type, 'train_baseline.py')
        script_path = Path(project_root) / "scripts" / script_name
        
        # 确保脚本存在
        if not script_path.exists():
            raise FileNotFoundError(f"训练脚本不存在: {script_path}")
            
        # 创建日志目录
        log_dir = Path(project_root) / "webapp" / "logs"
        log_dir.mkdir(exist_ok=True, parents=True)
        log_file = log_dir / f"training_{training_id}.log"
        
        # 构建训练命令
        cmd = [
            sys.executable,  # 当前Python解释器路径
            str(script_path),
            f"--epochs={config_data['epochs']}",
            f"--learning_rate={config_data['learning_rate']}",
            f"--experiment_name={model_type}_{training_id}"
        ]
        
        self.logger.info(f"启动训练: {' '.join(cmd)}")
        
        # 创建进度追踪文件
        progress_file = Path(project_root) / "logs" / f"progress_{training_id}.json"
        with open(progress_file, 'w') as f:
            json.dump({
                'status': 'starting',
                'progress': 0,
                'current_epoch': 0,
                'total_epochs': config_data['epochs'],
                'start_time': config_data['start_time']
            }, f)
        
        # 初始化状态
        self.training_status[training_id] = {
            'status': 'starting',
            'progress': 0,
            'current_epoch': 0,
            'total_epochs': config_data['epochs'],
            'train_loss': None,
            'val_loss': None,
            'train_acc': None,
            'val_acc': None,
            'start_time': config_data['start_time'],
            'message': '正在启动训练...',
            'log_file': str(log_file)
        }
        
        # 定义进程输出处理函数
        def process_output(proc):
            import re
            
            # 更新正则表达式，匹配实际的输出格式
            epoch_pattern = re.compile(r'Epoch\s+(\d+)/(\d+)')
            loss_acc_pattern = re.compile(r'Train Loss: ([\d\.]+), Train Acc: ([\d\.]+)% - Val Loss: ([\d\.]+), Val Acc: ([\d\.]+)%')
            best_acc_pattern = re.compile(r'Best validation accuracy: ([\d\.]+)%')
            
            # 打开日志文件
            with open(log_file, 'w', encoding='utf-8') as log_f:
                log_f.write(f"训练开始时间: {datetime.now().isoformat()}\n")
                log_f.write(f"训练命令: {' '.join(cmd)}\n\n")
                
                while True:
                    if proc.poll() is not None:  # 进程已结束
                        end_status = f"训练结束时间: {datetime.now().isoformat()}, 返回码: {proc.returncode}\n"
                        log_f.write(end_status)
                        
                        if proc.returncode == 0:
                            self.training_status[training_id]['status'] = 'completed'
                            self.training_status[training_id]['progress'] = 100
                            self.training_status[training_id]['message'] = '训练完成'
                            self.training_status[training_id]['end_time'] = datetime.now().isoformat()
                        else:
                            self.training_status[training_id]['status'] = 'failed'
                            self.training_status[training_id]['message'] = f'训练失败，返回码: {proc.returncode}'
                            self.training_status[training_id]['end_time'] = datetime.now().isoformat()
                        break
                    
                    line = proc.stdout.readline().decode('utf-8', errors='ignore').strip()
                    if not line:
                        continue
                    
                    # 将输出写入日志文件
                    log_f.write(line + '\n')
                    log_f.flush()
                    
                    # 解析训练进度
                    epoch_match = epoch_pattern.search(line)
                    if epoch_match:
                        current_epoch = int(epoch_match.group(1))
                        total_epochs = int(epoch_match.group(2))
                        progress = (current_epoch / total_epochs) * 100
                        
                        self.training_status[training_id]['current_epoch'] = current_epoch
                        self.training_status[training_id]['total_epochs'] = total_epochs
                        self.training_status[training_id]['progress'] = progress
                        self.training_status[training_id]['status'] = 'running'
                        self.training_status[training_id]['message'] = f'训练进行中... Epoch {current_epoch}/{total_epochs}'
                        
                        # 更新进度文件
                        with open(progress_file, 'w') as f:
                            json.dump({
                                'status': 'running',
                                'progress': progress,
                                'current_epoch': current_epoch,
                                'total_epochs': total_epochs
                            }, f)
                    
                    # 解析损失值和准确率
                    loss_acc_match = loss_acc_pattern.search(line)
                    if loss_acc_match:
                        train_loss = float(loss_acc_match.group(1))
                        train_acc = float(loss_acc_match.group(2))
                        val_loss = float(loss_acc_match.group(3))
                        val_acc = float(loss_acc_match.group(4))
                        
                        self.training_status[training_id]['train_loss'] = train_loss
                        self.training_status[training_id]['val_loss'] = val_loss
                        self.training_status[training_id]['train_acc'] = train_acc
                        self.training_status[training_id]['val_acc'] = val_acc
                    
                    # 解析最佳准确率
                    best_acc_match = best_acc_pattern.search(line)
                    if best_acc_match:
                        best_val_acc = float(best_acc_match.group(1))
                        self.training_status[training_id]['best_val_acc'] = best_val_acc
        
        # 启动进程
        try:
            # 设置环境变量，解决Unicode编码问题
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'  # 设置Python的IO编码为UTF-8
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=False,
                env=env  # 使用修改后的环境变量
            )
            
            self.training_processes[training_id] = process
            
            # 在后台线程中处理输出
            thread = threading.Thread(target=process_output, args=(process,))
            thread.daemon = True
            thread.start()
            
            self.logger.info(f"训练进程已启动，ID: {training_id}")
            return True
            
        except Exception as e:
            error_msg = f"启动训练失败: {str(e)}"
            self.logger.error(error_msg)
            self.training_status[training_id] = {
                'status': 'failed',
                'message': error_msg,
                'end_time': datetime.now().isoformat()
            }
            return False
    
    def _start_mock_training(self, training_id: str, config_data: Dict[str, Any]):
        """启动模拟训练进程（保留以便在实际训练无法进行时作为备选）"""
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
        if training_id in self.training_processes:
            try:
                self.training_processes[training_id].terminate()
                self.training_status[training_id]['status'] = 'stopped'
                self.training_status[training_id]['message'] = '训练已停止'
                return True
            except Exception as e:
                self.training_status[training_id]['message'] = f'停止训练失败: {str(e)}'
                return False
        
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