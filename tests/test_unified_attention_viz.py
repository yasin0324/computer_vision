#!/usr/bin/env python3
"""
统一注意力可视化脚本的单元测试
"""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch
import pandas as pd
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.unified_attention_visualization import UnifiedAttentionVisualizer
from src.config.config import Config


class TestUnifiedAttentionVisualizer(unittest.TestCase):
    """统一注意力可视化器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.config = Config()
        self.visualizer = UnifiedAttentionVisualizer(self.config)
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_visualizer_initialization(self):
        """测试可视化器初始化"""
        self.assertIsInstance(self.visualizer, UnifiedAttentionVisualizer)
        self.assertEqual(self.visualizer.config, self.config)
        self.assertEqual(len(self.visualizer.models), 0)
        self.assertIsNotNone(self.visualizer.class_names)
        self.assertIsNotNone(self.visualizer.label_to_idx)
    
    @patch('torch.load')
    def test_load_single_model_se_net(self, mock_torch_load):
        """测试SE-Net模型加载"""
        # 模拟torch.load返回值
        mock_checkpoint = {'model_state_dict': {}}
        mock_torch_load.return_value = mock_checkpoint
        
        # 模拟模型
        with patch('scripts.unified_attention_visualization.ResNetSE') as mock_model_class:
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            
            # 测试加载SE-Net模型
            model = self.visualizer._load_single_model('se_net', 'dummy_path.pth')
            
            # 验证模型类被正确调用
            mock_model_class.assert_called_once_with(
                num_classes=self.config.NUM_CLASSES,
                reduction=16,
                dropout_rate=0.7
            )
            
            # 验证模型方法被调用
            mock_model.model.load_state_dict.assert_called_once_with({})
            mock_model.to.assert_called_once()
            mock_model.eval.assert_called_once()
    
    @patch('torch.load')
    def test_load_single_model_cbam(self, mock_torch_load):
        """测试CBAM模型加载"""
        mock_checkpoint = {'model_state_dict': {}}
        mock_torch_load.return_value = mock_checkpoint
        
        with patch('scripts.unified_attention_visualization.ResNetCBAM') as mock_model_class:
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            
            model = self.visualizer._load_single_model('cbam', 'dummy_path.pth')
            
            mock_model_class.assert_called_once_with(
                num_classes=self.config.NUM_CLASSES,
                reduction=16,
                dropout_rate=0.7
            )
    
    def test_load_single_model_invalid_type(self):
        """测试无效模型类型"""
        with self.assertRaises(ValueError) as context:
            self.visualizer._load_single_model('invalid_model', 'dummy_path.pth')
        
        self.assertIn("不支持的模型类型", str(context.exception))
    
    def test_load_models_success(self):
        """测试批量模型加载成功"""
        model_configs = {
            'se_net_test': {'type': 'se_net', 'path': 'test_se.pth'},
            'cbam_test': {'type': 'cbam', 'path': 'test_cbam.pth'}
        }
        
        with patch.object(self.visualizer, '_load_single_model') as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model
            
            self.visualizer.load_models(model_configs)
            
            # 验证两个模型都被加载
            self.assertEqual(len(self.visualizer.models), 2)
            self.assertIn('se_net_test', self.visualizer.models)
            self.assertIn('cbam_test', self.visualizer.models)
    
    def test_load_models_with_error(self):
        """测试模型加载时出现错误"""
        model_configs = {
            'failing_model': {'type': 'se_net', 'path': 'nonexistent.pth'}
        }
        
        with patch.object(self.visualizer, '_load_single_model') as mock_load:
            mock_load.side_effect = Exception("File not found")
            
            # 应该不抛出异常，而是记录错误
            self.visualizer.load_models(model_configs)
            
            # 模型字典应该为空
            self.assertEqual(len(self.visualizer.models), 0)
    
    @patch('pandas.read_csv')
    @patch('scripts.unified_attention_visualization.TomatoSpotDataset')
    @patch('torch.utils.data.DataLoader')
    def test_create_data_loader(self, mock_dataloader, mock_dataset, mock_read_csv):
        """测试数据加载器创建"""
        # 模拟CSV读取
        mock_df = pd.DataFrame({'image_path': ['img1.jpg'], 'label': ['class1']})
        mock_read_csv.return_value = mock_df
        
        # 模拟数据集
        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance
        
        # 模拟数据加载器
        mock_loader = MagicMock()
        mock_dataloader.return_value = mock_loader
        
        # 测试数据加载器创建
        loader = self.visualizer._create_data_loader('test')
        
        # 验证CSV被读取
        mock_read_csv.assert_called_once()
        
        # 验证数据集被创建
        mock_dataset.assert_called_once()
        
        # 验证数据加载器被创建
        mock_dataloader.assert_called_once()
    
    def test_save_original_image(self):
        """测试原始图像保存"""
        # 创建测试张量
        image_tensor = torch.randn(1, 3, 224, 224)
        output_dir = Path(self.temp_dir)
        
        # 测试图像保存
        self.visualizer._save_original_image(image_tensor, output_dir)
        
        # 验证图像文件被创建
        image_path = output_dir / "original_image.png"
        self.assertTrue(image_path.exists())
    
    def test_extract_attention_weights(self):
        """测试注意力权重提取"""
        # 创建模拟模型
        mock_model = MagicMock()
        mock_model.model = MagicMock()
        
        # 创建模拟模块
        mock_se_module = MagicMock()
        mock_se_module.register_forward_hook = MagicMock()
        
        # 模拟named_modules返回
        mock_model.model.named_modules.return_value = [
            ('layer1.se', mock_se_module),
            ('layer2.conv', MagicMock())  # 这个不应该被注册
        ]
        
        # 创建测试输入
        input_tensor = torch.randn(1, 3, 224, 224)
        
        # 测试权重提取
        with patch.object(mock_model, '__call__') as mock_forward:
            mock_forward.return_value = torch.randn(1, 10)
            
            weights = self.visualizer._extract_attention_weights(mock_model, input_tensor)
            
            # 验证前向传播被调用
            mock_forward.assert_called_once_with(input_tensor)
            
            # 验证钩子被注册
            mock_se_module.register_forward_hook.assert_called_once()
    
    def test_generate_simple_report(self):
        """测试简单报告生成"""
        results = {
            'model1': {'pred_class': 0, 'confidence': 0.95},
            'model2': {'pred_class': 1, 'confidence': 0.82}
        }
        
        true_label = 0
        output_dir = Path(self.temp_dir)
        
        self.visualizer._generate_simple_report(results, true_label, output_dir)
        
        # 验证报告文件被创建
        report_path = output_dir / "simple_report.md"
        self.assertTrue(report_path.exists())
        
        # 验证报告内容
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('简单注意力分析报告', content)
            self.assertIn('model1', content)
            self.assertIn('model2', content)
            self.assertIn('0.95', content)
    
    def test_generate_comparison_report(self):
        """测试对比报告生成"""
        # 模拟模型预测
        mock_model1 = MagicMock()
        mock_model2 = MagicMock()
        
        self.visualizer.models = {
            'model1': mock_model1,
            'model2': mock_model2
        }
        
        # 模拟预测结果
        with patch('torch.no_grad'):
            mock_model1.return_value = torch.tensor([[2.0, 1.0, 0.5]])
            mock_model2.return_value = torch.tensor([[1.0, 2.0, 0.5]])
            
            image = torch.randn(1, 3, 224, 224)
            true_label = 0
            output_dir = Path(self.temp_dir)
            
            self.visualizer._generate_comparison_report(image, true_label, output_dir)
            
            # 验证报告文件被创建
            report_path = output_dir / "comparison_report.md"
            self.assertTrue(report_path.exists())
    
    def test_generate_comprehensive_analysis(self):
        """测试综合分析生成"""
        true_label = 0
        output_dir = Path(self.temp_dir)
        
        self.visualizer._generate_comprehensive_analysis(true_label, output_dir)
        
        # 验证分析文件被创建
        analysis_path = output_dir / "comprehensive_analysis.md"
        self.assertTrue(analysis_path.exists())
        
        # 验证内容包含注意力机制对比
        with open(analysis_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('SE-Net', content)
            self.assertIn('CBAM', content)
            self.assertIn('通道注意力', content)


class TestUnifiedVisualizationIntegration(unittest.TestCase):
    """集成测试类"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_models.json"
        
        # 创建测试配置文件
        test_config = {
            "test_se": {
                "type": "se_net",
                "path": "test_se.pth",
                "description": "Test SE-Net model"
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_file_loading(self):
        """测试配置文件加载"""
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        self.assertIn('test_se', config)
        self.assertEqual(config['test_se']['type'], 'se_net')
    
    @patch('sys.argv')
    @patch('scripts.unified_attention_visualization.UnifiedAttentionVisualizer')
    def test_main_function_call(self, mock_visualizer_class, mock_argv):
        """测试主函数调用"""
        # 模拟命令行参数
        mock_argv.__getitem__.side_effect = lambda x: [
            'script_name',
            '--models', str(self.config_path),
            '--sample_idx', '0',
            '--output_dir', self.temp_dir,
            '--mode', 'simple'
        ][x]
        
        mock_visualizer = MagicMock()
        mock_visualizer_class.return_value = mock_visualizer
        
        # 由于main函数中的argparse，我们需要模拟整个调用
        with patch('argparse.ArgumentParser.parse_args') as mock_parse_args:
            mock_args = MagicMock()
            mock_args.models = str(self.config_path)
            mock_args.sample_idx = 0
            mock_args.output_dir = self.temp_dir
            mock_args.mode = 'simple'
            mock_parse_args.return_value = mock_args
            
            with patch('scripts.unified_attention_visualization.Config') as mock_config:
                with patch('builtins.open', unittest.mock.mock_open(read_data='{"test": "config"}')):
                    with patch('json.load', return_value={"test": "config"}):
                        # 导入并调用main函数
                        from scripts.unified_attention_visualization import main
                        main()
                        
                        # 验证可视化器被创建和调用
                        mock_visualizer_class.assert_called_once()
                        mock_visualizer.load_models.assert_called_once()
                        mock_visualizer.visualize_sample.assert_called_once()


if __name__ == '__main__':
    unittest.main() 