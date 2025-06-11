#!/usr/bin/env python3
"""
植物叶片病害识别Web应用主文件
"""

import os
import sys
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import torch
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 简化导入，避免复杂依赖
try:
    from src.config.config import config
except ImportError:
    # 如果配置文件不存在，使用默认配置
    class DefaultConfig:
        TARGET_CLASSES = {
            0: 'bacterial_spot',
            1: 'healthy', 
            2: 'septoria_leaf_spot',
            3: 'target_spot'
        }
    config = DefaultConfig()

from webapp.utils import ModelPredictor, TrainingManager, FileManager

# 创建Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 配置上传文件夹
UPLOAD_FOLDER = Path(project_root) / 'webapp' / 'uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# 全局变量
model_predictor = None
training_manager = None
file_manager = None


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def init_managers():
    """初始化管理器"""
    global model_predictor, training_manager, file_manager
    
    try:
        model_predictor = ModelPredictor()
        training_manager = TrainingManager()
        file_manager = FileManager()
        print("✅ 管理器初始化成功")
    except Exception as e:
        print(f"❌ 管理器初始化失败: {e}")
        # 创建空的管理器以避免None错误
        model_predictor = type('MockPredictor', (), {
            'get_available_models': lambda: [],
            'predict_image': lambda *args: {'error': '模型未加载'}
        })()
        training_manager = type('MockTrainer', (), {
            'start_training': lambda *args: 'mock_id',
            'get_training_status': lambda *args: {'status': 'error', 'message': '训练器未初始化'}
        })()
        file_manager = type('MockFileManager', (), {
            'get_available_models': lambda: [],
            'get_dataset_info': lambda: {'error': '文件管理器未初始化'},
            'get_models_summary': lambda: {'total': 0, 'by_type': {}},
            'get_recent_evaluations': lambda *args: []
        })()


# 初始化管理器
init_managers()


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/predict')
def predict_page():
    """预测页面"""
    # 获取可用模型列表
    available_models = model_predictor.get_available_models()
    return render_template('predict.html', models=available_models)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """预测API"""
    try:
        # 检查是否有文件上传
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': '不支持的文件格式'}), 400
        
        # 获取模型类型
        model_type = request.form.get('model_type', 'baseline')
        
        # 保存上传的文件
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # 进行预测
        result = model_predictor.predict_image(file_path, model_type)
        
        # 清理临时文件
        os.remove(file_path)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/train')
def train_page():
    """训练页面"""
    return render_template('train.html')


@app.route('/api/train', methods=['POST'])
def api_train():
    """训练API"""
    try:
        data = request.get_json()
        
        # 获取训练参数
        model_type = data.get('model_type', 'baseline')
        epochs = data.get('epochs', 50)
        learning_rate = data.get('learning_rate', 0.0005)
        batch_size = data.get('batch_size', 32)
        
        # 启动训练
        training_id = training_manager.start_training(
            model_type=model_type,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size
        )
        
        return jsonify({
            'training_id': training_id,
            'message': '训练已开始',
            'status': 'started'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/training_status/<training_id>')
def api_training_status(training_id):
    """获取训练状态"""
    try:
        status = training_manager.get_training_status(training_id)
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/training_log/<training_id>')
def api_training_log(training_id):
    """获取训练日志"""
    try:
        status = training_manager.get_training_status(training_id)
        if 'log_file' not in status:
            return jsonify({'error': '训练日志不存在'}), 404
            
        log_file = status['log_file']
        if not os.path.exists(log_file):
            return jsonify({'error': '训练日志文件不存在'}), 404
            
        return send_file(log_file, mimetype='text/plain')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/training_log_content/<training_id>')
def api_training_log_content(training_id):
    """获取训练日志内容（支持增量获取）"""
    try:
        status = training_manager.get_training_status(training_id)
        if 'log_file' not in status:
            return jsonify({'error': '训练日志不存在'}), 404
            
        log_file = status['log_file']
        if not os.path.exists(log_file):
            return jsonify({'error': '训练日志文件不存在'}), 404
        
        # 获取偏移量参数
        offset = request.args.get('offset', default=0, type=int)
        
        # 读取日志文件
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            # 如果偏移量为0，返回全部内容
            if offset == 0:
                content = f.read()
                next_offset = len(content)
            else:
                # 如果偏移量大于0，先定位到偏移位置
                f.seek(offset)
                # 读取新增内容
                content = f.read()
                next_offset = offset + len(content)
        
        # 如果内容为空，返回空字符串
        if not content:
            return jsonify({
                'content': '',
                'next_offset': next_offset
            })
        
        # 处理ANSI转义序列（终端颜色等）
        # 这里简化处理，实际可能需要更复杂的ANSI到HTML的转换
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        content = ansi_escape.sub('', content)
        
        # 返回内容和新的偏移量
        return jsonify({
            'content': content,
            'next_offset': next_offset
        })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stop_training/<training_id>', methods=['POST'])
def api_stop_training(training_id):
    """停止训练"""
    try:
        result = training_manager.stop_training(training_id)
        if result:
            return jsonify({'message': '训练已停止'})
        else:
            return jsonify({'error': '停止训练失败'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/evaluate')
def evaluate_page():
    """评估页面"""
    # 获取可用模型列表
    available_models = file_manager.get_available_models()
    return render_template('evaluate.html', models=available_models)


@app.route('/api/evaluate', methods=['POST'])
def api_evaluate():
    """评估API - 真实评估版本"""
    try:
        data = request.get_json()
        
        model_path = data.get('model_path')
        model_type = data.get('model_type', 'baseline')
        
        if not model_path:
            return jsonify({'error': '请选择模型文件'}), 400
        
        model_file = Path(model_path)
        if not model_file.exists():
            return jsonify({'error': f'模型文件不存在: {model_path}'}), 400
            
        model_name = model_file.stem

        try:
            # 运行真实的模型评估
            eval_result = run_model_evaluation(model_path, model_type, model_name)
            
            # 检查评估结果中是否有错误
            if 'error' in eval_result:
                 return jsonify({'error': f"评估失败: {eval_result['error']}"}), 500

            # 创建输出目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(project_root) / "outputs" / "evaluation" / f"web_eval_{model_type}_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存评估结果
            results_file = output_dir / "evaluation_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(eval_result, f, indent=2, ensure_ascii=False)

            # 生成简单的HTML报告
            report_url = create_simple_html_report(eval_result, output_dir)

            return jsonify({
                'message': '评估完成',
                'results': {
                    'accuracy': eval_result.get('accuracy'),
                    'macro_f1': eval_result.get('f1_score'),
                    'output_dir': str(output_dir),
                    'report_url': report_url,
                    'class_metrics': eval_result.get('class_metrics'),
                    'confusion_matrix': eval_result.get('confusion_matrix'),
                    'class_names': eval_result.get('class_names')
                },
                'note': '当前为真实评估模式'
            })

        except Exception as e:
            return jsonify({'error': f'模型评估执行失败: {e}'}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def create_simple_html_report(results, output_dir):
    """根据评估结果生成一个简单的HTML报告"""
    model_name = results.get('model_name', 'N/A')
    model_type = results.get('model_type', 'N/A')
    
    # 构建HTML内容
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>模型评估报告 - {model_type}</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .metric {{ margin: 10px 0; }}
            .note {{ color: green; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>模型评估报告</h1>
        <div class="note">注意：当前为真实评估结果</div>
        <h2>基本信息</h2>
        <div class="metric">模型名称: {model_name}</div>
        <div class="metric">模型类型: {model_type}</div>
        <div class="metric">评估时间: {results.get('evaluation_date', 'N/A')}</div>
        
        <h2>性能指标</h2>
        <div class="metric">准确率: {results.get('accuracy', 0):.4f}</div>
        <div class="metric">宏平均F1: {results.get('f1_score', 0):.4f}</div>
        <div class="metric">精确率: {results.get('precision', 0):.4f}</div>
        <div class="metric">召回率: {results.get('recall', 0):.4f}</div>
    """
    
    # 添加类别指标
    if 'class_metrics' in results and results['class_metrics']:
        html_content += "<h2>各类别性能</h2><table border='1' style='border-collapse: collapse;'><tr><th>类别</th><th>精确率</th><th>召回率</th><th>F1分数</th></tr>"
        for class_name, metrics in results['class_metrics'].items():
            html_content += f"<tr><td>{class_name}</td><td>{metrics.get('precision', 0):.2f}</td><td>{metrics.get('recall', 0):.2f}</td><td>{metrics.get('f1', 0):.2f}</td></tr>"
        html_content += "</table>"
    
    html_content += "</body></html>"
    
    # 保存HTML文件
    html_file = output_dir / f"{model_type}_report.html"
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return f"/api/report/{html_file.name}"


@app.route('/compare')
def compare_page():
    """模型比较页面"""
    available_models = file_manager.get_available_models()
    return render_template('compare.html', models=available_models)


@app.route('/api/compare', methods=['POST'])
def api_compare():
    """模型比较API - 真实评估版本"""
    try:
        data = request.get_json()
        model_configs = data.get('models', [])
        
        if len(model_configs) < 2:
            return jsonify({'error': '至少需要选择两个模型进行比较'}), 400
        
        # 为每个模型运行真实评估
        comparison_results = []
        
        for i, config_item in enumerate(model_configs):
            model_name = config_item['name']
            model_type = config_item['type']
            model_path = config_item['path']
            
            print(f"正在评估模型 {i+1}/{len(model_configs)}: {model_name}")
            
            # 运行真实的模型评估
            try:
                eval_result = run_model_evaluation(model_path, model_type, model_name)
                comparison_results.append(eval_result)
                print(f"✅ {model_name} 评估完成")
            except Exception as e:
                print(f"❌ {model_name} 评估失败: {e}")
                # 如果评估失败，使用备用数据
                fallback_result = create_fallback_result(model_name, model_type, model_path, str(e))
                comparison_results.append(fallback_result)
        
        # 生成比较报告
        comparison_report = {
            'summary': {
                'total_models': len(comparison_results),
                'best_accuracy': max(r['accuracy'] for r in comparison_results if 'accuracy' in r),
                'best_f1': max(r['f1_score'] for r in comparison_results if 'f1_score' in r),
                'comparison_date': datetime.now().isoformat(),
                'evaluation_type': 'real_evaluation'
            },
            'models': comparison_results,
            'rankings': {
                'by_accuracy': sorted([r for r in comparison_results if 'accuracy' in r], 
                                    key=lambda x: x['accuracy'], reverse=True),
                'by_f1_score': sorted([r for r in comparison_results if 'f1_score' in r], 
                                    key=lambda x: x['f1_score'], reverse=True),
                'by_efficiency': sorted([r for r in comparison_results if 'inference_time_ms' in r], 
                                      key=lambda x: x['inference_time_ms'])
            }
        }
        
        # 保存比较报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(project_root) / "outputs" / "evaluation" / f"web_comparison_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = output_dir / "comparison_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_report, f, indent=2, ensure_ascii=False)
        
        return jsonify({
            'message': '模型比较完成',
            'comparison_report': comparison_report,
            'charts_dir': str(output_dir),
            'note': '使用真实模型评估结果进行比较'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def run_model_evaluation(model_path, model_type, model_name):
    """运行真实的模型评估"""
    import subprocess
    
    # 使用独立的评估脚本
    eval_script_path = Path(__file__).parent / "eval_script.py"
    
    try:
        # 运行评估脚本
        result = subprocess.run(
            [sys.executable, str(eval_script_path), model_path, model_type, model_name],
            capture_output=True,
            text=True,
            timeout=300,  # 5分钟超时
            cwd=str(project_root),
            env=dict(os.environ, PYTHONPATH=str(project_root))
        )
        
        if result.returncode == 0:
            # 解析结果
            stdout_text = result.stdout.strip()
            
            # 查找JSON行（通常是以{开头的行）
            json_line = None
            for line in stdout_text.split('\n'):
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    json_line = line
                    break
            
            if json_line:
                eval_result = json.loads(json_line)
                return eval_result
            else:
                raise Exception(f"未找到有效JSON输出。完整输出: {stdout_text}")
        else:
            # 获取详细错误信息
            error_msg = f"返回码: {result.returncode}\n"
            if result.stdout:
                error_msg += f"标准输出: {result.stdout}\n"
            if result.stderr:
                error_msg += f"错误输出: {result.stderr}"
            print(f"详细错误信息: {error_msg}")
            raise Exception(f"评估脚本执行失败: {error_msg}")
            
    except Exception as e:
        raise e


def create_fallback_result(model_name, model_type, model_path, error_msg):
    """创建备用结果（当评估失败时）"""
    model_file = Path(model_path)
    file_size_mb = model_file.stat().st_size / (1024*1024) if model_file.exists() else 0
    
    return {
        'model_name': model_name,
        'model_type': model_type,
        'model_path': model_path,
        'accuracy': 0.0,
        'f1_score': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'file_size_mb': round(file_size_mb, 1),
        'inference_time_ms': 0.0,
        'evaluation_date': datetime.now().isoformat(),
        'status': f'评估失败: {error_msg}',
        'error': True
    }


@app.route('/api/report/<filename>')
def api_report(filename):
    """获取报告文件"""
    try:
        # 查找报告文件
        reports_dir = Path(project_root) / "outputs" / "evaluation"
        
        # 递归查找文件
        for report_file in reports_dir.rglob(filename):
            return send_file(report_file)
        
        return jsonify({'error': '报告文件未找到'}), 404
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models')
def api_models():
    """获取可用模型列表"""
    try:
        models = file_manager.get_available_models()
        return jsonify(models)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets')
def api_datasets():
    """获取数据集信息"""
    try:
        dataset_info = file_manager.get_dataset_info()
        return jsonify(dataset_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/dashboard')
def dashboard():
    """仪表板页面"""
    return render_template('dashboard.html')


@app.route('/api/dashboard_data')
def api_dashboard_data():
    """获取仪表板数据"""
    try:
        # 获取模型统计信息
        models_info = file_manager.get_models_summary()
        
        # 获取最近的评估结果
        recent_evaluations = file_manager.get_recent_evaluations(limit=5)
        
        # 获取数据集统计
        dataset_stats = file_manager.get_dataset_stats()
        
        return jsonify({
            'models_info': models_info,
            'recent_evaluations': recent_evaluations,
            'dataset_stats': dataset_stats,
            'system_info': {
                'gpu_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'current_time': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found_error(error):
    """404错误处理"""
    return render_template('error.html', error_code=404, error_message="页面未找到"), 404


@app.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    return render_template('error.html', error_code=500, error_message="服务器内部错误"), 500


if __name__ == '__main__':
    # 初始化管理器
    init_managers()
    
    # 运行应用
    app.run(debug=True, host='0.0.0.0', port=5000) 